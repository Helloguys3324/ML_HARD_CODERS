#!/usr/bin/env python3
"""task_china solver — oracle-bone glyph counting -> 8-letter word.

Approach (mirrors baseline_solution.py contract):
  1. For each crate glyph, average its labelled train boxes into a template.
  2. Slide the template (normalised cross-correlation) over every rubbing of
     every set, count non-maximum-suppressed peaks above a threshold.
  3. Write counts.csv (crate, set, glyph_id, count) and decode the letter.

Usage: .venv/bin/python challenges/task_china/solve.py --data challenges/task_china/data
"""
from __future__ import annotations

import argparse
import json
import string
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

TEMPLATE = 40
THRESHOLD = 0.45
MAX_BOXES = 300


def build_template(data: Path, labels: pd.DataFrame, glyph_id: str, max_boxes: int = MAX_BOXES):
    rows = labels[labels.glyph_id == glyph_id].head(max_boxes)
    crops = []
    wts, hts = [], []
    for name, group in rows.groupby("image"):
        img = cv2.imread(str(data / "train" / "images" / name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        for r in group.itertuples():
            x, y, w, h = int(r.x), int(r.y), int(r.w), int(r.h)
            crop = img[y:y + h, x:x + w]
            if crop.size:
                crops.append(cv2.resize(crop, (TEMPLATE, TEMPLATE)).astype(np.float32))
                wts.append(w)
                hts.append(h)
    template = np.mean(crops, axis=0)
    return template.astype(np.uint8), float(np.median(wts)), float(np.median(hts))


def count_matches_score(img: np.ndarray, template: np.ndarray, w: float, h: float, thr: float):
    """Return (count, list_of_scores) for NMS peaks above thr."""
    fx, fy = TEMPLATE / w, TEMPLATE / h
    small = cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
    if small.shape[0] <= TEMPLATE or small.shape[1] <= TEMPLATE:
        return 0, np.array([])
    score = cv2.matchTemplate(small, template, cv2.TM_CCOEFF_NORMED)
    kernel = np.ones((TEMPLATE, TEMPLATE), np.uint8)
    local_max = score == cv2.dilate(score, kernel)
    peaks = local_max & (score >= thr)
    return int(peaks.sum()), score[peaks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("challenges/task_china/data"))
    ap.add_argument("--out", type=Path, default=Path("challenges/task_china/counts.csv"))
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--scores", action="store_true", help="dump per-set peak score stats")
    a = ap.parse_args()

    labels = pd.read_csv(a.data / "train_labels.csv")
    crates = pd.read_csv(a.data / "crates.csv")
    print(f"{labels.image.nunique()} labelled rubbings, {len(labels)} boxes, {labels.glyph_id.nunique()} ids")

    rows = []
    stats = {}
    for crate, glyph_id in zip(crates.crate, crates.glyph_id):
        template, w, h = build_template(a.data, labels, glyph_id)
        per_set = []
        all_scores = []
        for set_dir in sorted((a.data / "crates" / crate).iterdir()):
            if not set_dir.is_dir():
                continue
            n = 0
            sc = []
            for f in sorted(set_dir.glob("*.jpg")):
                img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                cnt, scores = count_matches_score(img, template, w, h, a.threshold)
                n += cnt
                sc.extend(scores.tolist())
            per_set.append(n)
            all_scores.append(sc)
            rows.append([crate, set_dir.name, glyph_id, n])
        print(crate, glyph_id, "counts_per_set:", per_set)
        stats[crate] = {"glyph_id": glyph_id, "counts": per_set}
        if a.scores:
            flat = [s for sc in all_scores for s in sc]
            if flat:
                arr = np.array(flat)
                print(f"    peak_scores: n={len(flat)} min={arr.min():.3f} "
                      f"p25={np.percentile(arr,25):.3f} med={np.median(arr):.3f} "
                      f"p75={np.percentile(arr,75):.3f} max={arr.max():.3f}")

    df = pd.DataFrame(rows, columns=["crate", "set", "glyph_id", "count"])
    df.to_csv(a.out, index=False)
    print("written", a.out)

    # --- decoding attempts ---
    print("\n=== DECODING ===")
    counts_by_crate = {}
    for crate, glyph_id in zip(crates.crate, crates.glyph_id):
        counts_by_crate[crate] = stats[crate]["counts"]
    json.dump(counts_by_crate, open(Path(a.out).with_suffix(".json"), "w"), indent=2)

    def a1z26(n):
        n = int(round(n))
        if 1 <= n <= 26:
            return string.ascii_uppercase[n - 1]
        return None

    print("sum -> letter:", end=" ")
    for crate in crates.crate:
        s = sum(counts_by_crate[crate])
        L = a1z26(s)
        print(f"{crate}:{s}{('=' + L) if L else ''}", end="  ")
    print()
    print("per-set values (to eyeball):")
    for crate in crates.crate:
        print(f"  {crate}: {counts_by_crate[crate]}")


if __name__ == "__main__":
    main()