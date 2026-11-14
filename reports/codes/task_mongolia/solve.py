#!/usr/bin/env python3
"""
LMML 2026 — task_mongolia ("The Baa-rilliant task!")
Count sheep in drone survey photos; every plot's count reveals a word -> flag.

Pipeline:
  1. Extract per-image features (light/dark blobs, peaks, saturation,...).
  2. Calibrate a regressor on the 280 training photos (exact counts known).
  3. Predict counts for the 333 survey photos, save counts.csv.
  4. Decode the word from counts + plot positions -> SIGMOID_<WORD>.
"""
from __future__ import annotations
import os
import numpy as np
import cv2
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "data")
TRAIN = os.path.join(DATA, "train")
SURVEY = os.path.join(DATA, "survey")
SIGMA = 16


def load_image(path: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def features_from_image(im: np.ndarray) -> dict:
    f = {}
    imf = im.astype(np.float32)
    L = imf.mean(axis=2)
    mx = imf.max(axis=2); mn = imf.min(axis=2)
    S = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0)

    smooth = cv2.GaussianBlur(L, (0, 0), SIGMA)
    c = L - smooth                      # local contrast: + bright, - dark
    ac = np.abs(c)

    f["L_mean"] = L.mean()
    f["L_std"] = L.std()
    f["S_mean"] = S.mean()
    f["ac_mean"] = ac.mean()
    f["ac_p99"] = np.percentile(ac, 99)

    # bright blobs
    for thr in (18, 30, 45):
        mk = (c > thr).astype(np.uint8)
        mk = cv2.morphologyEx(mk, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, st, _ = cv2.connectedComponentsWithStats(mk, connectivity=8)
        areas = st[1:, 4]
        big = areas[areas >= 12]
        f[f"nb_{thr}"] = len(big)
        f[f"ab_{thr}"] = int(big.sum()) if len(big) else 0

    # dark blobs
    for thr in (18, 30):
        mk = (c < -thr).astype(np.uint8)
        mk = cv2.morphologyEx(mk, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, st, _ = cv2.connectedComponentsWithStats(mk, connectivity=8)
        areas = st[1:, 4]
        big = areas[areas >= 12]
        f[f"nd_{thr}"] = len(big)
        f[f"ad_{thr}"] = int(big.sum()) if len(big) else 0

    # local maxima of bright contrast (instance-ish peak count)
    peakmap = np.where(c > 18, c, 0).astype(np.float32)
    dmap = cv2.distanceTransform((c < 12).astype(np.uint8), cv2.DIST_L2, 5)
    f["dmap_mean"] = dmap.mean()
    f["peak_sum"] = peakmap.sum()
    return f


def extract(df: pd.DataFrame, folder: str):
    feats = []
    for img in df["image"]:
        im = load_image(os.path.join(folder, img))
        feats.append(features_from_image(im))
    F = pd.DataFrame(feats)
    F.insert(0, "image", df["image"].values)
    return F


def main():
    tc = pd.read_csv(os.path.join(DATA, "train_counts.csv"))
    tc = tc.rename(columns={"count": "true"})
    F = extract(tc, TRAIN)
    F["true"] = tc["true"].values
    F.to_csv(os.path.join(os.path.dirname(__file__), "train_features.csv"), index=False)
    print("train features saved", F.shape)

    # quick correlation of each feature vs true
    corr = F.drop(columns=["image", "true"]).corrwith(F["true"])
    print(corr.sort_values(ascending=False).round(3))


if __name__ == "__main__":
    main()