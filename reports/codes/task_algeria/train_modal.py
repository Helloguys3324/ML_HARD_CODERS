"""Modal GPU LoRA training for task_algeria (SmolLM2-360M-Instruct register SFT).

Usage:
    /home/kozak/mllm/py -m modal run train_modal.py --epochs 4 --rank 32

The dataset (train.jsonl, built by build_dataset.py) is passed as a string.
The PEFT adapter and the GGUF adapter are written to the lmml-data volume:
    /data/task_algeria/adapter_peft/...
    /data/task_algeria/adapter.gguf
Then download with:
    /home/kozak/mllm/py -m modal volume get lmml-data task_algeria ./modal_out
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
APP = "task-algeria-lora"
BASE = "HuggingFaceTB/SmolLM2-360M-Instruct"
SYSTEM = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

data_vol = modal.Volume.from_name("lmml-data", create_if_missing=True)
cache_vol = modal.Volume.from_name("lmml-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .uv_pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "peft==0.13.2",
        "accelerate==1.1.1",
        "safetensors",
        "sentencepiece",
        "numpy",
        "gguf",
        "huggingface_hub[hf_xet]",
        "hf_transfer",
    )
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)

app = modal.App(APP, image=image)


def build_prompt(user: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    )


@app.function(gpu="A10G", timeout=7200, volumes={"/data": data_vol, "/cache": cache_vol})
def train(
    jsonl: str,
    epochs: float = 4.0,
    rank: int = 32,
    alpha: int = 64,
    lr: float = 2e-4,
    batch_size: int = 16,
    grad_accum: int = 1,
    max_len: int = 256,
    dropout: float = 0.05,
    seed: int = 20261114,
    out_name: str = "run1",
    convert: bool = True,
) -> dict:
    import random

    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    start = time.time()
    rows = [json.loads(line) for line in jsonl.splitlines() if line.strip()]
    print(f"dataset: {len(rows)} pairs", flush=True)
    random.seed(seed)
    torch.manual_seed(seed)

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=TARGETS,
            task_type="CAUSAL_LM",
            bias="none",
        ),
    )
    model.print_trainable_parameters()
    model = model.cuda()
    model.train()

    examples = []
    for row in rows:
        prompt = build_prompt(row["prompt"])
        answer = row["reply"] + "<|im_end|>\n"
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        ids = tok(prompt + answer, add_special_tokens=False).input_ids
        if len(ids) > max_len:
            continue
        labels = [-100] * len(p_ids) + ids[len(p_ids):]
        examples.append((ids, labels))
    print(f"encoded {len(examples)} examples (max_len={max_len})", flush=True)

    pad_id = tok.pad_token_id
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    n_ex = len(examples)
    steps_per_epoch = max(1, n_ex // (batch_size * grad_accum))
    total_steps = max(1, int(steps_per_epoch * epochs))
    warmup = min(50, total_steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (s + 1) / max(1, warmup) if s < warmup
        else max(0.02, 0.5 * (1 + np.cos(np.pi * (s - warmup) / max(1, total_steps - warmup)))),
    )

    def make_batch(idx):
        chunk = [examples[i] for i in idx]
        L = max(len(x) for x, _ in chunk)
        ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
        lab = torch.full((len(chunk), L), -100, dtype=torch.long)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for j, (x, y) in enumerate(chunk):
            ids[j, : len(x)] = torch.tensor(x)
            lab[j, : len(y)] = torch.tensor(y)
            att[j, : len(x)] = 1
        return ids, lab, att

    order = list(range(n_ex))
    step = 0
    losses = []
    for epoch in range(int(epochs) if float(epochs).is_integer() else int(epochs) + 1):
        random.shuffle(order)
        for start_i in range(0, n_ex - batch_size + 1, batch_size):
            idx = order[start_i:start_i + batch_size]
            ids, lab, att = make_batch(idx)
            ids, lab, att = ids.cuda(), lab.cuda(), att.cuda()
            out = model(input_ids=ids, attention_mask=att, labels=lab)
            loss = out.loss / grad_accum
            loss.backward()
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            losses.append(float(out.loss))
            step += 1
            if step % 25 == 0:
                print(f"step {step}/{total_steps} loss {sum(losses[-25:]) / len(losses[-25:]):.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {time.time()-start:.0f}s", flush=True)
            if step >= total_steps:
                break
        if step >= total_steps:
            break
    print(f"training done: {step} steps, mean loss {sum(losses[-100:]) / max(1,len(losses[-100:])):.4f}", flush=True)

    out_dir = Path("/data/task_algeria") / f"adapter_peft_{out_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=True)
    tok.save_pretrained(str(out_dir))
    data_vol.commit()
    print("saved", out_dir, flush=True)

    gguf_path = None
    if convert:
        # ship the exact converter next to the adapter (located from the mounted repo copy)
        import shutil
        import subprocess

        conv = Path("/data/task_algeria/tools/lora_to_gguf.py")
        if conv.is_file():
            gguf_path = Path("/data/task_algeria") / f"adapter_{out_name}.gguf"
            r = subprocess.run(
                ["python", str(conv), str(out_dir), "--outfile", str(gguf_path)],
                capture_output=True, text=True,
            )
            print(r.stdout[-1000:], r.stderr[-1000:], flush=True)
            if r.returncode != 0:
                gguf_path = None
        data_vol.commit()
    return {
        "run": out_name, "examples": len(examples), "steps": step,
        "seconds": round(time.time() - start, 1),
        "adapter_dir": str(out_dir),
        "gguf": str(gguf_path) if gguf_path else None,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


@app.local_entrypoint()
def main(
    epochs: float = 4.0,
    rank: int = 32,
    alpha: int = 64,
    lr: float = 2e-4,
    batch_size: int = 16,
    max_len: int = 256,
    out_name: str = "run1",
    convert: bool = True,
):
    jsonl = (HERE / "train.jsonl").read_text(encoding="utf-8")
    print(f"uploading dataset: {len(jsonl)/1024:.0f} KiB, {len(jsonl.splitlines())} lines")
    result = train.remote(
        jsonl=jsonl, epochs=epochs, rank=rank, alpha=alpha, lr=lr,
        batch_size=batch_size, max_len=max_len, out_name=out_name, convert=convert,
    )
    print(json.dumps(result, indent=2))
