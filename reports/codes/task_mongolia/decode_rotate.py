#!/usr/bin/env python3
"""Decode: render counts in the survey's natural (rotated) frame as ASCII art."""
import os
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "data")


def rotated_frame(sp, counts):
    df = sp.merge(counts, on="image", how="left")
    df["count"] = df["count"].fillna(0)
    x = df["x_m"].values; y = df["y_m"].values
    pts = np.stack([x, y], 1)
    mu = pts.mean(0)
    c = np.cov(pts.T)
    ev, evec = np.linalg.eigh(c)
    maj = evec[:, np.argmax(ev)]      # long axis
    minor = evec[:, np.argmin(ev)]    # cross-track
    long_ = (pts - mu) @ maj
    short_ = (pts - mu) @ minor
    df["long_"] = long_; df["short_"] = short_
    return df, long_, short_


def render(df, cell_l, cell_s, transform="value"):
    l, s, v = df["long_"].values, df["short_"].values, df["count"].values
    lmin, lmax = l.min(), l.max()
    smin, smax = s.min(), s.max()
    ncol = int(np.ceil((lmax - lmin) / cell_l)) + 1
    nrow = int(np.ceil((smax - smin) / cell_s)) + 1
    g = np.zeros((nrow, ncol)); cnt = np.zeros((nrow, ncol))
    for li, si, vi in zip(l, s, v):
        c_idx = int((li - lmin) / cell_l)
        r = int((smax - si) / cell_s)
        if 0 <= r < nrow and 0 <= c_idx < ncol:
            g[r, c_idx] += vi; cnt[r, c_idx] += 1
    chars = " .:-=+*#%@"
    nch = len(chars)
    lines = []
    for r in range(nrow):
        line = ""
        for c_idx in range(ncol):
            if cnt[r, c_idx] == 0:
                line += " "
                continue
            val = g[r, c_idx]
            if transform == "mean":
                val = val / cnt[r, c_idx]
            elif transform == "presence":
                line += "#" if val > 0 else "."
                continue
            idx = min(nch - 1, int(round(val)))
            line += chars[idx]
        lines.append(line)
    return lines, g, cnt


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("counts")
    ap.add_argument("--cell_l", type=float, default=50)
    ap.add_argument("--cell_s", type=float, default=50)
    ap.add_argument("--transform", default="mean", choices=["value", "mean", "presence"])
    args = ap.parse_args()
    sp = pd.read_csv(os.path.join(DATA, "survey_plots.csv"))
    counts = pd.read_csv(args.counts)
    df, l, s = rotated_frame(sp, counts)
    lines, g, cnt = render(df, args.cell_l, args.cell_s, args.transform)
    print(f"=== transform={args.transform} cell_l={args.cell_l} cell_s={args.cell_s} grid {len(lines)}x{len(lines[0])} ===")
    for ln in lines:
        print(ln)
    print(f"long range {l.min():.0f}..{l.max():.0f} ({l.max()-l.min():.0f}m) | short range {s.min():.0f}..{s.max():.0f} ({s.max()-s.min():.0f}m)")