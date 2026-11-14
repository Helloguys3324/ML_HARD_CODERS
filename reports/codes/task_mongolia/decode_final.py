#!/usr/bin/env python3
"""Decode the census word from survey counts.

Renders (plot position -> count) as ASCII art in the survey strip's natural
frame. Supports several value transforms and grid resolutions.
"""
import os
import numpy as np
import pandas as pd

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
    return df


def render(df, cell_l=25, cell_s=25, transform="value", vmin=None, vmax=None, quantile_norm=True):
    L, S, v = df["L"].values, df["S"].values, df["count"].values
    Lmin, Lmax = L.min(), L.max(); Smin, Smax = S.min(), S.max()
    ncol = int(np.ceil((Lmax - Lmin) / cell_l)) + 1
    nrow = int(np.ceil((Smax - Smin) / cell_s)) + 1
    acc = np.zeros((nrow, ncol)); cnt = np.zeros((nrow, ncol))
    for li, si, vi in zip(L, S, v):
        cc = int((li - Lmin) / cell_l); rr = int((Smax - si) / cell_s)
        if 0 <= rr < nrow and 0 <= cc < ncol:
            acc[rr, cc] += vi; cnt[rr, cc] += 1
    if transform == "mean":
        cellval = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    elif transform == "sum":
        cellval = np.where(cnt > 0, acc, np.nan)
    else:
        cellval = np.where(cnt > 0, acc, np.nan)

    valid = cellval[~np.isnan(cellval)]
    lo = np.nanmin(valid); hi = np.nanmax(valid)
    if quantile_norm:
        lo = np.percentile(valid, 2); hi = np.percentile(valid, 98)
    chars = " .:-=+*#%@"
    nch = len(chars)
    lines = []
    for r in range(nrow):
        line = ""
        for cc in range(ncol):
            x = cellval[r, cc]
            if np.isnan(x):
                line += " "
                continue
            z = (x - lo) / max(1e-6, hi - lo)
            idx = int(np.clip(round(z * (nch - 1)), 0, nch - 1))
            line += chars[idx]
        lines.append(line)
    return lines


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("counts")
    ap.add_argument("--cell_l", type=float, default=25)
    ap.add_argument("--cell_s", type=float, default=25)
    ap.add_argument("--transform", default="sum", choices=["sum", "mean", "interp"])
    args = ap.parse_args()
    df = load(args.counts)
    lines = render(df, args.cell_l, args.cell_s, args.transform)
    print(f"=== {args.transform} cell_l={args.cell_l} cell_s={args.cell_s} (grid {len(lines)}x{len(lines[0])}) ===")
    for ln in lines:
        print(ln)