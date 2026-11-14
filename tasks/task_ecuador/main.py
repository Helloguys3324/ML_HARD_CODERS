#!/usr/bin/env python3
"""Ciclopaseo Mirage -- bounded (max +/-1 per RGB value) targeted attack.

The script is executed offline as ``python main.py`` in a directory containing
``test-input.png``, ``challenge_model.py`` and
``classifier/mobilenet_v3_large-5c1a4163.pth``.  It writes ``test-output.png``:
RGB, same width/height as the input, at most 1 MiB, and classified as ImageNet
class 665 (moped) with softmax confidence >= 0.99.

Method: targeted projected gradient descent with a straight-through estimator
on an integer delta map.  Every pixel/channel may move by at most one integer
step (-1, 0 or +1) from its original value.  The classifier is the very same
checkpoint used by the evaluator, so no transfer surrogate is needed.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from challenge_model import (
    MAX_OUTPUT_BYTES,
    MIN_TARGET_CONFIDENCE,
    TARGET_CLASS_ID,
    load_model,
    normalize_batch,
    pil_to_tensor,
    predict_tensor,
)

# Stop attacking once we are comfortably above the required 0.99 threshold.
TARGET_CONFIDENCE = 0.997
# Wall-clock budget (seconds from process start) for the whole attack loop.
# The platform allows 60 s; leave room for model load, PNG encoding and a
# final check on the saved PNG.
TIME_BUDGET_S = 44.0
# Safety cap on gradient iterations (the time budget normally binds first).
MAX_ITERS = 120
# PGD step in integer units (continuous relaxation, rounded for every forward).
STEP = 0.5
# Momentum on the sign gradient (MI-FGSM style).
MOMENTUM = 0.9


def _round_away_from_zero(d: torch.Tensor) -> torch.Tensor:
    """Round continuous deltas to {-1, 0, +1}; ties go away from zero."""
    return torch.sign(d) * torch.floor(d.abs() + 0.5)


def _to_int_delta(d: torch.Tensor) -> torch.Tensor:
    return _round_away_from_zero(d).clamp(-1.0, 1.0)


def _apply_delta(pixels: np.ndarray, delta: torch.Tensor) -> np.ndarray:
    """Apply an integer CHW delta in {-1,0,+1} to an HWC uint8 image."""
    d = delta.detach().cpu().numpy().astype(np.int16)
    if d.ndim == 4:
        d = d[0]
    d = np.transpose(d, (1, 2, 0))
    return np.clip(pixels.astype(np.int16) + d, 0, 255).astype(np.uint8)


def _save_png(pixels: np.ndarray, path: str) -> None:
    """Save an RGB PNG, retrying with maximum compression if it is too large."""
    image = Image.fromarray(pixels, mode="RGB")
    image.save(path, format="PNG", compress_level=6)
    if os.path.getsize(path) > MAX_OUTPUT_BYTES:
        image.save(path, format="PNG", compress_level=9, optimize=True)


def _attack(
    model: torch.nn.Module,
    pixels: np.ndarray,
    *,
    deadline: float,
    target_confidence: float = TARGET_CONFIDENCE,
    max_iters: int = MAX_ITERS,
) -> tuple[torch.Tensor, float, int]:
    """Targeted PGD on an integer-valued +/-1 perturbation.

    Returns (best CHW delta, best confidence, iterations run).
    """
    x0 = (
        torch.from_numpy(pixels.astype(np.int32))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
    )
    target = torch.tensor([TARGET_CLASS_ID])

    # --- Initial forward/backward at the untouched image -------------------
    d = torch.zeros_like(x0).requires_grad_(True)
    logits = model(normalize_batch((x0 + d) / 255.0))
    best_conf = float(logits.softmax(dim=1)[0, TARGET_CLASS_ID].item())
    best_delta = torch.zeros_like(x0)
    if best_conf >= target_confidence:
        return best_delta, best_conf, 0

    loss = F.cross_entropy(logits, target)
    g = torch.autograd.grad(loss, d)[0]
    g_acc = g / (g.abs().mean() + 1e-12)
    with torch.no_grad():
        # First-order warm start: one integer FGSM step against the target CE.
        d = (-torch.sign(g_acc)).clamp(-1.0, 1.0)
    d.requires_grad_(True)

    iters = 0
    while iters < max_iters and time.monotonic() < deadline:
        iters += 1
        di = _to_int_delta(d)
        # Straight-through estimator: forward uses the integer delta, the
        # gradient is taken as if rounding were the identity.
        x_int = (x0 + (d + (di - d).detach())).clamp(0.0, 255.0)
        logits = model(normalize_batch(x_int / 255.0))
        conf = float(logits.softmax(dim=1)[0, TARGET_CLASS_ID].item())
        if conf > best_conf:
            best_conf = conf
            best_delta = di.detach().clone()
        if conf >= target_confidence:
            break

        loss = F.cross_entropy(logits, target)
        g = torch.autograd.grad(loss, d)[0]
        g_acc = MOMENTUM * g_acc + g / (g.abs().mean() + 1e-12)
        with torch.no_grad():
            # Descend the targeted cross-entropy in L-inf integer space.
            d = (d - STEP * torch.sign(g_acc)).clamp(-1.0, 1.0)
        d.requires_grad_(True)

    return best_delta, best_conf, iters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="test-input.png")
    parser.add_argument("--output", default="test-output.png")
    parser.add_argument(
        "--model",
        default=os.path.join("classifier", "mobilenet_v3_large-5c1a4163.pth"),
    )
    parser.add_argument("--time-budget", type=float, default=TIME_BUDGET_S)
    parser.add_argument("--target-conf", type=float, default=TARGET_CONFIDENCE)
    args = parser.parse_args()

    # The evaluator provides 2 CPU cores.
    torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    t_start = time.monotonic()
    model = load_model(args.model)

    with Image.open(args.input) as image:
        image = image.convert("RGB")
    pixels = np.array(image, dtype=np.uint8)
    height, width = pixels.shape[:2]

    deadline = t_start + args.time_budget
    delta, best_conf, iters = _attack(
        model,
        pixels,
        deadline=deadline,
        target_confidence=args.target_conf,
    )

    out = _apply_delta(pixels, delta)
    _save_png(out, args.output)

    # --- Verify the PNG that will actually be scored ----------------------
    with Image.open(args.output) as saved:
        saved = saved.convert("RGB")
        saved_pixels = np.array(saved, dtype=np.int16)
        prediction = predict_tensor(model, pil_to_tensor(saved))
    max_delta = int(np.abs(saved_pixels - pixels.astype(np.int16)).max())
    size_ok = os.path.getsize(args.output) <= MAX_OUTPUT_BYTES
    shape_ok = saved.size == (width, height)
    pass_ok = (
        prediction.class_id == TARGET_CLASS_ID
        and prediction.target_confidence >= MIN_TARGET_CONFIDENCE
        and max_delta <= 1
        and size_ok
        and shape_ok
    )
    print(
        f"input={args.input} size={width}x{height} iters={iters} "
        f"attack_conf={best_conf:.6f} saved_conf={prediction.target_confidence:.6f} "
        f"class={prediction.class_id} max_delta={max_delta} "
        f"bytes={os.path.getsize(args.output)} pass={pass_ok} "
        f"elapsed={time.monotonic() - t_start:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
