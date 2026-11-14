#!/usr/bin/env python3
"""Comprehensive decoder: try many encodings to extract the census word."""
import os
import numpy as np
import pandas as pd
from scipy.interpolate import griddata, NearestNDInterpolator

DATA = os.path.join(os.path.dirname(__file__), "data")


def load(counts_path):
    sp = pd.read_csv(os.path.join(DATA, "survey_plots.csv"))
    c = pd.read_csv(counts_path)
    df = sp.merge(c, on="image", how="left")
    df["count"] = df["count"].fillna(0).astype(float)
    x = df["x_m"].values; y = df["y_m"].values
    pts = np.stack([x, y], 1)
    mu = pts.mean(0); cov = np.cov(pts.T); ev, evec = np.linalg.eigh(cov)
    maj = evec[:, np.argmax(ev)]; minor = evec[:, np.argmin(ev)]
    df["L"] = (pts - mu) @ maj
    df["S"] = (pts - mu) @ minor
    df["x"] = x; df["y"] = y
    return df


def interp(df, nx, ny, zcol="count"):
    L, S, v = df["L"].values, df["S"].values, df[zcol].values
    Lg = np.linspace(L.min(), L.max(), nx); Sg = np.linspace(S.min(), S.max(), ny)
    GG, HH = np.meshgrid(Lg, Sg)
    V = griddata(np.stack([L, S], 1), v, (GG, HH), method="linear", fill_value=np.nan)
    Vn = NearestNDInterpolator(np.stack([L, S], 1), v)(GG, HH)
    return np.where(np.isnan(V), Vn, V)


def ascii_art(V, nlevels=10, qlo=2, qhi=98):
    chars = " .:-=+*#%@"
    lo = np.percentile(V, qlo); hi = np.percentile(V, qhi)
    ny, nx = V.shape
    for r in range(ny):
        line = ""
        for c in range(nx):
            z = (V[r, c] - lo) / max(1e-6, hi - lo)
            i = int(np.clip(round(z * (len(chars) - 1)), 0, len(chars) - 1))
            line += chars[i]
        print(line)


def binary_art(V, thr):
    ny, nx = V.shape
    for r in range(ny):
        print("".join("#" if V[r, c] > thr else "." for c in range(nx)))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("counts")
    ap.add_argument("--nx", type=int, default=74)
    ap.add_argument("--ny", type=int, default=16)
    args = ap.parse_args()
    df = load(args.counts)
    V = interp(df, args.nx, args.ny)
    print(f"=== grayscale (sum counts, {args.nx}x{args.ny}) ===")
    ascii_art(V)
    for q in (0.6, 0.7, 0.8):
        thr = np.percentile(df['count'], q * 100)
        print(f"\n=== binary count > {thr:.0f} (top {(1-q)*100:.0f}%) ===")
        binary_art(V, thr)