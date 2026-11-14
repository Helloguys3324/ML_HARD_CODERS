"""
Train a PEFT LoRA adapter on SmolLM2-360M-Instruct (CPU) to speak the Call of the Square
register. Produces an adapter folder; convert it with data/tools/lora_to_gguf.py.

Everything runs on CPU. A 360M model training on a ~64-token context with batch 8 is
feasible in tens of minutes on 20 cores; keep r small and steps low.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

TASK = Path(__file__).resolve().parent
DATA = TASK / "data"
sys.path.insert(0, str(DATA))

import torch

SYSTEM = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"


def build_prompt(user: str) -> str:
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    ap.add_argument("--corpus", default=str(TASK / "corpus.jsonl"))
    ap.add_argument("--out", default=str(TASK / "adapter"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--grad-checkpoint", action="store_true", default=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = use whole corpus")
    ap.add_argument("--threads", type=int, default=20)
    return ap.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    corpus = [json.loads(ln) for ln in open(args.corpus, encoding="utf-8") if ln.strip()]
    if args.limit:
        random.shuffle(corpus)
        corpus = corpus[: args.limit]
    print(f"corpus: {len(corpus)} pairs")

    tok = AutoTokenizer.from_pretrained(args.base)
    print("loading base model (CPU, fp32) ...")
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32)
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()

    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    model.train()

    def encode(q: str, a: str):
        prompt = build_prompt(q)
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        ids = tok(prompt + a + "<|im_end|>\n", add_special_tokens=False).input_ids
        return ids, [-100] * len(p_ids) + ids[len(p_ids):]

    data = [encode(q, a) for q, a in ((c["q"], c["a"]) for c in corpus)]
    # drop over-long samples
    data = [(x, y) for x, y in data if len(x) <= args.max_len]
    print(f"usable samples: {len(data)}")
    pad = tok.pad_token_id

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)

    steps_per_epoch = (len(data) + args.batch - 1) // args.batch
    total_steps = steps_per_epoch * args.epochs
    print(f"{steps_per_epoch} steps/epoch, {total_steps} total steps, batch {args.batch}")

    idx = list(range(len(data)))
    t0 = time.time()
    global_step = 0
    for epoch in range(args.epochs):
        random.shuffle(idx)
        for bi in range(0, len(idx), args.batch):
            batch_idx = idx[bi:bi + args.batch]
            batch = [data[i] for i in batch_idx]
            L = max(len(x) for x, _ in batch)
            ids = torch.full((len(batch), L), pad, dtype=torch.long)
            lab = torch.full((len(batch), L), -100, dtype=torch.long)
            att = torch.zeros((len(batch), L), dtype=torch.long)
            for j, (x, y) in enumerate(batch):
                ids[j, :len(x)] = torch.tensor(x, dtype=torch.long)
                lab[j, :len(y)] = torch.tensor(y, dtype=torch.long)
                att[j, :len(x)] = 1
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            global_step += 1
            if global_step % 10 == 0:
                dt = time.time() - t0
                print(f"  step {global_step}/{total_steps} loss {loss.item():.3f} "
                      f"({dt:.0f}s, {dt / max(1, global_step):.2f}s/step)", flush=True)

    model.save_pretrained(args.out)
    print(f"saved adapter -> {args.out}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()