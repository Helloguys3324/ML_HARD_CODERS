#!/usr/bin/env python3
"""Detect + count a target glyph in crate images with the trained CNN.

Uses a sliding window (mirroring baseline_solution.py) but with softmax
confidence from cnn.pt. Supports a --validate mode that evaluates the detector
on the labelled train images so we can calibrate the threshold.

Usage:
  .venv/bin/python challenges/task_china/detect.py --validate --threshold 0.5
  .venv/bin/python challenges/task_china/detect.py --threshold 0.5
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from train_cnn import Net

D = Path("challenges/task_china/data")
OUT = Path("challenges/task_china")
SIZE = 48
STRIDE = 4


def load_model():
    meta = json.load(open(OUT / "classes.json"))
    g2i, glyphs = meta["g2i"], meta["glyphs"]
    model = Net(len(glyphs))
    model.load_state_dict(torch.load(OUT / "cnn.pt", map_location="cpu"))
    model.eval()
    return model, g2i


def build_heatmap(img_gray, model, class_idx, med_w, med_h, stride=STRIDE):
    """Return (count, heatmap) — count = number of NMS peaks above a per-call prob threshold."""
    fx, fy = SIZE / med_w, SIZE / med_h
    small = cv2.resize(img_gray, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
    H, W = small.shape
    if H <= SIZE or W <= SIZE:
        return 0, None, small
    # sliding window probability map
    ys = list(range(0, H - SIZE + 1, stride))
    xs = list(range(0, W - SIZE + 1, stride))
    # batch all patches
    grid = np.zeros((H, W), dtype=np.float32)
    # process in tiles to bound memory
    probs = np.full((H, W), -1.0, dtype=np.float32)
    windows = []
    pos = []
    for y in ys:
        for x in xs:
            windows.append(small[y:y + SIZE, x:x + SIZE])
            pos.append((y, x))
    windows = np.stack(windows).astype(np.float32) / 255.0
    with torch.no_grad():
        for s in range(0, len(windows), 256):
            xb = torch.from_numpy(windows[s:s + 256])
            xb = xb[:, None]
            out = model(xb)
            p = F.softmax(out, 1)[:, class_idx].numpy()
            for k, pv in enumerate(p):
                y, x = pos[s + k]
                # accumulate max prob per pixel (stride tiles)
                probs[y:y + SIZE, x:x + SIZE] = np.maximum(probs[y:y + SIZE, x:x + SIZE], pv)
    return probs, small


def nms_count(probs, threshold, img_gray=None):
    if probs is None:
        return 0, np.array([])
    kernel = np.ones((SIZE, SIZE), np.uint8)
    local_max = probs == cv2.dilate(probs, kernel)
    peaks = local_max & (probs >= threshold)
    return int(peaks.sum()), probs[peaks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--out", default=str(OUT / "counts_cnn.csv"))
    a = ap.parse_args()

    model, g2i = load_model()
    labels = pd.read_csv(D / "train_labels.csv")
    med_w = float(np.median(labels.w))
    med_h = float(np.median(labels.h))
    print(f"median box {med_w:.1f}x{med_h:.1f}", flush=True)

    if a.validate:
        # evaluate on train images: compare detected count vs true box count
        crates = pd.read_csv(D / "crates.csv")
        for crate, glyph_id in zip(crates.crate, crates.glyph_id):
            ci = g2i[glyph_id]
            sub = labels[labels.glyph_id == glyph_id]
            imgs = sub.image.unique()
            tp = fp = fn = 0
            t0 = time.time()
            for name in imgs[:120]:
                img = cv2.imread(str(D / "train" / "images" / name), cv2.IMREAD_GRAYSCALE)
                probs, _ = build_heatmap(img, model, ci, med_w, med_h)
                pred, _ = nms_count(probs, a.threshold)
                true = int((sub.image == name).sum())
                # matched via counting diff
                tp += min(pred, true)
                fp += max(0, pred - true)
                fn += max(0, true - pred)
            print(f"  {crate} {glyph_id}: thr={a.threshold}  (on {len(imgs[:120])} imgs) "
                  f"pred={tp+fp} true={tp+fn}  tp={tp} fp={fp} fn={fn} "
                  f"prec={tp/max(1,tp+fp):.3f} rec={tp/max(1,tp+fn):.3f}  ({time.time()-t0:.0f}s)", flush=True)
        return

    # production: count in crate sets
    crates = pd.read_csv(D / "crates.csv")
    rows = []
    for crate, glyph_id in zip(crates.crate, crates.glyph_id):
        ci = g2i[glyph_id]
        per_set = []
        for set_dir in sorted((D / "crates" / crate).iterdir()):
            if not set_dir.is_dir():
                continue
            n = 0
            for f in sorted(set_dir.glob("*.jpg")):
                img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                probs, _ = build_heatmap(img, model, ci, med_w, med_h)
                cnt, _ = nms_count(probs, a.threshold)
                n += cnt
            per_set.append(n)
            rows.append([crate, set_dir.name, glyph_id, n])
        print(crate, glyph_id, "counts_per_set:", per_set, flush=True)

    df = pd.DataFrame(rows, columns=["crate", "set", "glyph_id", "count"])
    df.to_csv(a.out, index=False)
    print("written", a.out)


if __name__ == "__main__":
    main()