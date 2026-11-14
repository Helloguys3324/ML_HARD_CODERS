#!/usr/bin/env python3
"""Calibrate template-matching counting on labelled train images (ground truth).

For each crate glyph, build a bank of individual 40x40 templates and measure
matchTemplate (TM_CCOEFF_NORMED) scores at real glyph locations (positives)
vs other glyph locations (negatives). This reveals whether simple template
matching (with alignment, as the baseline does) can separate glyphs well.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

D = Path("challenges/task_china/data")
OUT = Path("challenges/task_china")
SIZE = 40


def build_templates(labels, gid, n=40):
    rows = labels[labels.glyph_id == gid].sample(min(n, (labels.glyph_id == gid).sum()), random_state=3)
    tmpl = []
    cache = {}
    for r in rows.itertuples():
        p = D / "train" / "images" / r.image
        if p not in cache:
            cache[p] = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        c = cache[p][int(r.y):int(r.y + r.h), int(r.x):int(r.x + r.w)]
        if c.size:
            tmpl.append(cv2.resize(c, (SIZE, SIZE)).astype(np.float32))
    return tmpl


def score_at(img_gray, templates, x, y, w, h, med_w, med_h):
    """Best matchTemplate score of any template in a window around this box."""
    fx, fy = SIZE / med_w, SIZE / med_h
    small = cv2.resize(img_gray, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
    cx = fx * (x + w / 2)
    cy = fy * (y + h / 2)
    half = SIZE * 1.5
    x0 = int(max(0, cx - half)); y0 = int(max(0, cy - half))
    x1 = int(min(small.shape[1], cx + half)); y1 = int(min(small.shape[0], cy + half))
    if x1 - x0 < SIZE or y1 - y0 < SIZE:
        return None
    region = small[y0:y1, x0:x1].astype(np.float32)
    best = -1.0
    for t in templates:
        res = cv2.matchTemplate(region, t, cv2.TM_CCOEFF_NORMED)
        m = float(res.max())
        if m > best:
            best = m
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glyph", default="g015")
    a = ap.parse_args()
    labels = pd.read_csv(D / "train_labels.csv")
    med_w = float(np.median(labels.w)); med_h = float(np.median(labels.h))
    import json
    crates = pd.read_csv(D / "crates.csv")
    gmap = dict(zip(crates.glyph_id, crates.crate))

    for gid in ["g015", "g001", "g003", "g010"]:
        tmpl = build_templates(labels, gid, n=30)
        pos = labels[labels.glyph_id == gid].sample(min(100, (labels.glyph_id == gid).sum()), random_state=0)
        neg = labels[~labels.glyph_id.isin([gid, "rare"])].sample(150, random_state=0)
        cache = {}
        def imgof(name):
            p = D / "train" / "images" / name
            if p not in cache: cache[p] = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            return cache[p]
        pos_sc = []; neg_sc = []
        t0 = time.time()
        for r in pos.itertuples():
            s = score_at(imgof(r.image), tmpl, r.x, r.y, r.w, r.h, med_w, med_h)
            if s is not None: pos_sc.append(s)
        for r in neg.itertuples():
            s = score_at(imgof(r.image), tmpl, r.x, r.y, r.w, r.h, med_w, med_h)
            if s is not None: neg_sc.append(s)
        pos_sc = np.array(pos_sc); neg_sc = np.array(neg_sc)
        print(f"{gid} ({gmap.get(gid)}): pos n={len(pos_sc)} mean={pos_sc.mean():.3f} p10={np.percentile(pos_sc,10):.3f} med={np.median(pos_sc):.3f}")
        print(f"        neg n={len(neg_sc)} mean={neg_sc.mean():.3f} p90={np.percentile(neg_sc,90):.3f} p99={np.percentile(neg_sc,99):.3f}")
        for thr in [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
            rec = (pos_sc >= thr).mean(); fpr = (neg_sc >= thr).mean()
            print(f"        thr={thr}: recall={rec:.3f} fpr={fpr:.4f}")
        print(f"        ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()