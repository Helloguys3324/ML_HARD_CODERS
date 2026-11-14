#!/usr/bin/env python3
"""Backup counter: handcrafted features + GBM, for decoding if CNN is delayed."""
import os
import numpy as np
import cv2
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "data")


def feats(path):
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(float)
    L = im.mean(2)
    mx = im.max(2); mn = im.min(2)
    S = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0)
    f = {}
    f["L_mean"] = L.mean(); f["L_std"] = L.std(); f["S_mean"] = S.mean()
    f["L_p90"] = np.percentile(L, 90); f["L_p99"] = np.percentile(L, 99)
    smooth = cv2.GaussianBlur(L, (0, 0), 16)
    c = L - smooth
    f["ac_mean"] = np.abs(c).mean(); f["ac_p99"] = np.percentile(np.abs(c), 99)
    for thr in (18, 30, 45):
        mk = (c > thr).astype(np.uint8)
        mk = cv2.morphologyEx(mk, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, st, _ = cv2.connectedComponentsWithStats(mk, connectivity=8)
        a = st[1:, 4]; big = a[a >= 12]
        f[f"nb_{thr}"] = len(big); f[f"ab_{thr}"] = int(big.sum())
        mk2 = (c < -thr).astype(np.uint8)
        mk2 = cv2.morphologyEx(mk2, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n2, _, st2, _ = cv2.connectedComponentsWithStats(mk2, connectivity=8)
        a2 = st2[1:, 4]; big2 = a2[a2 >= 12]
        f[f"nd_{thr}"] = len(big2); f[f"ad_{thr}"] = int(big2.sum())
    # peaks on distance transform
    binary = (c > 18).astype(np.uint8)
    dt = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    from scipy import ndimage
    for mind in (6, 9):
        mx_ = ndimage.maximum_filter(dt, size=mind)
        peaks = (dt == mx_) & (dt > mind * 0.5)
        f[f"pk_{mind}"] = int(peaks.sum())
    return f


def extract(images, folder):
    rows = []
    for img in images:
        f = feats(os.path.join(folder, img))
        f["image"] = img
        rows.append(f)
    return pd.DataFrame(rows)


def main():
    tc = pd.read_csv(os.path.join(DATA, "train_counts.csv"))
    if os.path.exists(os.path.join(os.path.dirname(__file__), "train_feats2.csv")):
        F = pd.read_csv(os.path.join(os.path.dirname(__file__), "train_feats2.csv"))
    else:
        F = extract(tc["image"].tolist(), os.path.join(DATA, "train"))
        F["true"] = tc["count"].values
        F.to_csv(os.path.join(os.path.dirname(__file__), "train_feats2.csv"), index=False)

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import KFold
    X = F.drop(columns=["image", "true"]).values
    y = F["true"].values
    kf = KFold(5, shuffle=True, random_state=0)
    errs, preds, tt = [], [], []
    for tr, te in kf.split(X):
        m = GradientBoostingRegressor(n_estimators=400, max_depth=3, random_state=0)
        m.fit(X[tr], y[tr]); p = m.predict(X[te])
        errs.append(np.abs(p - y[te]).mean()); preds.append(p); tt.append(y[te])
    p = np.concatenate(preds); yy = np.concatenate(tt)
    print("GBM MAE %.2f RMSE %.2f r2 %.3f" % (np.mean(errs), np.sqrt(np.mean((p - yy) ** 2)), 1 - np.var(yy - p) / np.var(yy)))

    sp = pd.read_csv(os.path.join(DATA, "survey_plots.csv"))
    S = extract(sp["image"].tolist(), os.path.join(DATA, "survey"))
    m = GradientBoostingRegressor(n_estimators=400, max_depth=3, random_state=0)
    m.fit(X, y)
    cnt = np.maximum(0, m.predict(S.drop(columns=["image"]).values)).round().astype(int)
    out = pd.DataFrame({"image": sp["image"].values, "count": cnt})
    out.to_csv(os.path.join(os.path.dirname(__file__), "counts_v2.csv"), index=False)
    print("wrote counts_v2.csv; mean", cnt.mean().round(1), "sum", cnt.sum(), "zeros", (cnt == 0).sum())


if __name__ == "__main__":
    main()