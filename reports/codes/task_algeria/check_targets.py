"""Offline sanity check: score the plain reading of every training target for the
public prompts under the untouched base model, exactly like evaluation_script's
log-prob gate.  This tells us whether the corpus's deregistered replies are
plausible English *before* spending a GPU run.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DATA = Path("/home/kozak/mllm/MML/work/tasks/task_algeria/data")
sys.path.insert(0, str(DATA))
from style_checker import deregister, load_vocab  # noqa: E402
from evaluation_script import build_prompt, mean_logprob  # noqa: E402

BASE = "HuggingFaceTB/SmolLM2-360M-Instruct"


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)
    model.eval()
    v = load_vocab()
    pairs = [json.loads(line) for line in (HERE / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    target = {}
    for p in pairs:
        target.setdefault(p["prompt"], p["reply"])
    public = json.loads((DATA / "public" / "prompts.json").read_text(encoding="utf-8"))

    def score(user, reply):
        prompt_ids = tok(build_prompt(user), add_special_tokens=False).input_ids
        reply_ids = tok(reply, add_special_tokens=False).input_ids
        if not reply_ids:
            return None
        ids = torch.tensor([prompt_ids + reply_ids])
        with torch.no_grad():
            logits = model(ids).logits[0].float()
        total = 0.0
        for pos in range(len(prompt_ids), len(ids[0])):
            row = logits[pos - 1].numpy()
            m = row.max()
            lse = m + math.log(float(np.exp(row - m).sum()))
            total += float(row[ids[0, pos]]) - float(lse)
        return total / len(reply_ids)

    scores = []
    for p in public:
        rep = target.get(p["prompt"])
        if rep is None:
            print("missing", p["id"])
            continue
        plain = deregister(rep, v)
        lp = score(p["prompt"], plain)
        scores.append(lp)
        if lp is not None and lp < -2.6:
            print(f"LOW {lp:.2f} {p['id']}: {plain}")
    print(f"n={len(scores)} mean={sum(scores)/len(scores):.3f} min={min(scores):.3f}")


if __name__ == "__main__":
    main()
