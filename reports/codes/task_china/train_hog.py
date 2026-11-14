#!/usr/bin/env python3
"""Light multi-class glyph classifier (HOG + raw pixels, sklearn) for task_china.

Trains fast on CPU, saves model. Used as a cheap first pass to reveal the count
encoding; escalate to CNN if counts look too noisy.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog

D = Path("challenges/task_china/data")
OUT = Path("challenges/task_china")
SIZE = 48


def featurize(X: np.ndarray) -> np.ndarray:
    n = len(X)
    F = np.zeros((n, SIZE * SIZE + 324), dtype=np.float32)
    for i in range(n):
        img = X[i].astype(np.float32) / 255.0
        F[i, :SIZE * SIZE] = img.ravel()
        h = hog(img, orientations=9, pixels_per_cell=(12, 12), cells_per_block=(2, 2),
                feature_vector=True)
        F[i, SIZE * SIZE:] = h
    return F


def main():
    t0 = time.time()
    z = np.load(OUT / "crops.npz")
    X, y = z["X"], z["y"]
    meta = json.load(open(OUT / "classes.json"))
    g2i, glyphs = meta["g2i"], meta["glyphs"]
    n_cls = len(glyphs)

    # stratified split
    from collections import Counter
    tr, va = [], []
    rng = np.random.RandomState(0)
    for c in range(n_cls):
        ii = np.where(y == c)[0]
        rng.shuffle(ii)
        nv = max(1, int(0.15 * len(ii)))
        va.extend(ii[:nv]); tr.extend(ii[nv:])
    tr = np.array(tr); va = np.array(va)
    print(f"featurizing {len(X)}...", flush=True)
    F = featurize(X)
    print(f"features done {F.shape} ({time.time()-t0:.0f}s)", flush=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    Ftr = sc.fit_transform(F[tr]); Fva = sc.transform(F[va])
    clf = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs")
    clf.fit(Ftr, y[tr])
    print(f"trained ({time.time()-t0:.0f}s)", flush=True)

    acc = clf.score(Fva, y[va])
    print(f"top-1 val acc = {acc:.4f}", flush=True)

    # per target precision/recall
    P = clf.predict_proba(Fva)
    targets = ["g015", "g028", "g005", "g018", "g003", "g011", "g001", "g010"]
    for g in targets:
        ci = g2i[g]
        mask = y[va] == ci
        top1 = P.argmax(1) == ci
        tp = (top1 & mask).sum()
        rec = tp / max(1, mask.sum())
        prec = tp / max(1, top1.sum())
        print(f"  {g}: recall={rec:.3f} precision={prec:.3f} "
              f"prob_pos_mean={P[mask,ci].mean():.3f} prob_neg_mean={P[~mask,ci].mean():.3f}", flush=True)

    import pickle
    pickle.dump({"scaler": sc, "clf": clf}, open(OUT / "hog_clf.pkl", "wb"))
    print("saved hog_clf.pkl", flush=True)


if __name__ == "__main__":
    main()