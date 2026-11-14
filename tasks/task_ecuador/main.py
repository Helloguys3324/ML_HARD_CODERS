#!/usr/bin/env python3
"""Ciclopaseo Mirage — task_ecuador solver.

White-box L-infinity attack (eps = 1/255 == +-1 in uint8 space) that flips a
bicycle photo to ImageNet class 665 ("moped") with softmax confidence >= 0.99.

Strategy
--------
The evaluator runs the *exact same* MobileNetV3-Large (IMAGENET1K_V2)
checkpoint we receive, so this is a pure white-box attack (no transferability
needed, no network access required).

We may only move every RGB value by at most 1 on the 0-255 scale, i.e. an
epsilon of 1/255 in [0,1] space.  In this tiny ball the logits are near-linear
in the perturbation, so sign(gradient) is already a strong attack; we then
refine with momentum iterative FGSM (MI-FGSM) on the continuous relaxation
`delta in [-eps, eps]`, and finally round the perturbation to integer
{-1, 0, +1} per channel (relative to the ORIGINAL image).

The attack is wall-clock budget aware so it always finishes safely inside the
60 s CPU timeout, whatever the test image's resolution.
"""
from __future__ import annotations

import argparse
import time

_PROCESS_START = time.time()  # as early as possible: account for import time too

from pathlib import Path

import torch
from PIL import Image

from challenge_model import (
    TARGET_CLASS_ID,
    load_model,
    normalize_batch,
    pil_to_tensor,
)

MODEL_PATH = "classifier/mobilenet_v3_large-5c1a4163.pth"
EPS = 1.0 / 255.0            # +-1 uint8 per channel
MOMENTUM = 0.9
ATTACK_DEADLINE = 50.0       # hard absolute deadline (s from process start) for the
                             # optimization; save + exit happen right after, inside 60 s
MAX_STEPS = 600              # hard cap against slow CPU; 100 steps already reach
                             # 0.999 on the train image


def target_confidence(model: torch.nn.Module, u8: torch.Tensor) -> float:
    """Softmax probability of the target class for a (1,3,H,W) uint8-range tensor."""
    x01 = u8.div(255.0)
    logits = model(normalize_batch(x01))
    return torch.softmax(logits, dim=1)[0, TARGET_CLASS_ID].item()


def mi_fgsm(model, x0, target, steps, init_delta=None, g_acc=None, deadline=None):
    """Run MI-FGSM targeting `target`, returning (delta, g_acc) in [0,1] units.

    `x0` is (1,3,H,W) float in [0,1].  `delta` stays clamped to [-EPS, EPS].
    Stops early once `deadline` (monotonic seconds) is reached.
    """
    if init_delta is None:
        delta = torch.zeros_like(x0).uniform_(-EPS, EPS).clamp_(-EPS, EPS)
    else:
        delta = init_delta.clone()
    delta.requires_grad_(True)
    if g_acc is None:
        g_acc = torch.zeros_like(x0)

    for _ in range(steps):
        if deadline is not None and time.time() > deadline:
            break
        xadv = (x0 + delta).clamp(0.0, 1.0)
        logits = model(normalize_batch(xadv))
        loss = torch.nn.functional.cross_entropy(logits, target)
        grad = torch.autograd.grad(loss, delta)[0]
        g_acc = MOMENTUM * g_acc + grad
        with torch.no_grad():
            delta.data -= EPS * torch.sign(g_acc)
            delta.data.clamp_(-EPS, EPS)

    return delta.detach(), g_acc


def round_to_u8(x0: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Clamp + round the perturbed image to integer RGB (as float tensor)."""
    return (x0 + delta).clamp(0.0, 1.0).mul(255.0).round().clamp(0.0, 255.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="test-input.png")
    parser.add_argument("--output", default="test-output.png")
    args = parser.parse_args()

    torch.set_num_threads(2)

    model = load_model(MODEL_PATH)
    source = Image.open(args.input).convert("RGB")
    x0 = pil_to_tensor(source)              # (1,3,H,W) float in [0,1]
    target = torch.tensor([TARGET_CLASS_ID])

    # --- measure the per-step cost so we can plan the budget ---
    t0 = time.time()
    delta = torch.zeros_like(x0).uniform_(-EPS, EPS).clamp_(-EPS, EPS).requires_grad_(True)
    g_acc = torch.zeros_like(x0)
    for _ in range(4):
        xadv = (x0 + delta).clamp(0.0, 1.0)
        logits = model(normalize_batch(xadv))
        loss = torch.nn.functional.cross_entropy(logits, target)
        grad = torch.autograd.grad(loss, delta)[0]
        g_acc = MOMENTUM * g_acc + grad
        with torch.no_grad():
            delta.data -= EPS * torch.sign(g_acc)
            delta.data.clamp_(-EPS, EPS)
    per_step = (time.time() - t0) / 4.0

    deadline = _PROCESS_START + ATTACK_DEADLINE
    total_steps = max(int((deadline - time.time()) / per_step), 80)
    total_steps = min(total_steps, MAX_STEPS)

    # --- restarted MI-FGSM, keeping the best rounded result ---
    restarts = 3 if total_steps >= 240 else (2 if total_steps >= 120 else 1)
    steps_per = max(1, total_steps // restarts)

    best_u8 = None
    best_conf = -1.0
    # first restart from the clean zero point (most consistent), rest randomly
    delta, g_acc = mi_fgsm(
        model, x0, target, steps_per, init_delta=torch.zeros_like(x0), deadline=deadline
    )
    u8 = round_to_u8(x0, delta)
    conf = target_confidence(model, u8)
    if conf > best_conf:
        best_conf = conf
        best_u8 = u8
    for _ in range(restarts - 1):
        delta, g_acc = mi_fgsm(model, x0, target, steps_per, deadline=deadline)
        u8 = round_to_u8(x0, delta)
        conf = target_confidence(model, u8)
        if conf > best_conf:
            best_conf = conf
            best_u8 = u8

    assert best_u8 is not None

    # --- save as integer RGB PNG ---
    array = best_u8[0].to(torch.uint8).permute(1, 2, 0).numpy()
    im = Image.fromarray(array, mode="RGB")
    im.save(args.output, format="PNG")

    # --- round-trip: reopen the saved file and re-check like the evaluator ---
    with Image.open(args.output) as recheck:
        recheck.load()
        saved_size = recheck.size
        saved_mode = recheck.mode
        saved_format = recheck.format
    saved = pil_to_tensor(Image.open(args.output).convert("RGB")).mul(255.0)
    saved_conf = target_confidence(model, saved)

    src_u8 = x0.mul(255.0).round().clamp(0.0, 255.0)
    max_delta = int((saved - src_u8).abs().max().item())
    out_bytes = Path(args.output).stat().st_size

    print(
        f"moped confidence(saved)={saved_conf:.6f}  max_pixel_delta={max_delta}  "
        f"bytes={out_bytes}  size={saved_size}  mode={saved_mode}  fmt={saved_format}"
    )


if __name__ == "__main__":
    main()