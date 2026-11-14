#!/usr/bin/env python3
"""Development solver for task_ecuador ("Ciclopaseo Mirage").

Reproduces the white-box L-infinity attack used in `main.py`, but against the
*public* `data/train-input.png` and with full instrumentation of the scoring
criteria.  Run from the challenge directory:

    .venv/bin/python solve.py

The real submission is `main.py` (same attack, adaptive/robust, reads
`test-input.png`, writes `test-output.png`).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DATA = Path(__file__).resolve().parent / "data"
sys.path.insert(0, str(DATA))

from challenge_model import (  # noqa: E402
    TARGET_CLASS_ID,
    load_model,
    normalize_batch,
    pil_to_tensor,
)

MODEL_PATH = DATA / "classifier" / "mobilenet_v3_large-5c1a4163.pth"
EPS = 1.0 / 255.0
MOMENTUM = 0.9


def conf(model, u8):
    with torch.no_grad():
        p = torch.softmax(model(normalize_batch(u8.div(255.0))), dim=1)[0]
    return p.argmax().item(), p[TARGET_CLASS_ID].item()


def attack(model, x0, target, steps):
    delta = torch.zeros_like(x0)
    delta.requires_grad_(True)
    g_acc = torch.zeros_like(x0)
    for _ in range(steps):
        xadv = (x0 + delta).clamp(0.0, 1.0)
        logits = model(normalize_batch(xadv))
        loss = torch.nn.functional.cross_entropy(logits, target)
        grad = torch.autograd.grad(loss, delta)[0]
        g_acc = MOMENTUM * g_acc + grad
        with torch.no_grad():
            delta.data -= EPS * torch.sign(g_acc)
            delta.data.clamp_(-EPS, EPS)
    return (x0 + delta).clamp(0.0, 1.0).mul(255.0).round().clamp(0.0, 255.0)


def main():
    torch.set_num_threads(2)
    model = load_model(MODEL_PATH)

    src = Image.open(DATA / "train-input.png").convert("RGB")
    x0 = pil_to_tensor(src)
    target = torch.tensor([TARGET_CLASS_ID])
    src_u8 = (x0 * 255.0).round().clamp(0.0, 255.0)

    # sweep a few step budgets to illustrate convergence
    for steps in (1, 40, 100, 300):
        t0 = time.time()
        u8 = attack(model, x0, target, steps)
        cid, c = conf(model, u8)
        md = int((u8 - src_u8).abs().max().item())
        print(
            f"steps={steps:4d}  top={cid}  moped_conf={c:.6f}  "
            f"max_pixel_delta={md}  ({time.time()-t0:.2f}s)"
        )

    # final: 300 steps -> save PNG and verify the artifact
    u8 = attack(model, x0, target, 300)
    cid, c = conf(model, u8)
    out = Path(__file__).resolve().parent / "final-output.png"
    arr = u8[0].to(torch.uint8).permute(1, 2, 0).numpy()
    Image.fromarray(arr, mode="RGB").save(out, format="PNG")

    reloaded = pil_to_tensor(Image.open(out).convert("RGB")).mul(255.0)
    md = int((reloaded - src_u8).abs().max().item())
    ok = (
        cid == TARGET_CLASS_ID
        and c >= 0.99
        and md <= 1
        and out.stat().st_size <= 1024 * 1024
    )
    print(
        f"\nSaved {out.name}: predicted_class={cid} ({'moped' if cid==TARGET_CLASS_ID else '?'}) "
        f"confidence={c:.6f} max_pixel_delta={md} bytes={out.stat().st_size} -> {'PASS' if ok else 'FAIL'}"
    )


if __name__ == "__main__":
    main()