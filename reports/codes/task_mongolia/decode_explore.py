#!/usr/bin/env python3
"""Explore how survey plot counts + positions encode the flag word."""
import os
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "data")


def load_counts(path: str) -> pd.DataFrame:
    return pd.read_csv(path)  # image,count


def render_grid(df, xcol, ycol, vcol, cell, transform='value', xflip=False, yflip=False):
    x = df[xcol].values; y = df[ycol].values; v = df[vcol].values
    xmin, xmax = x.min(), x.max(); ymin, ymax = y.min(), y.max()
    ncol = int(np.ceil((xmax - xmin) / cell)) + 1
    nrow = int(np.ceil((ymax - ymin) / cell)) + 1
    g = np.zeros((nrow, ncol))
    for xi, yi, vi in zip(x, y, v):
        c = int((xi - xmin) / cell)
        r = int((ymax - yi) / cell)
        if 0 <= r < nrow and 0 <= c < ncol:
            g[r, c] += vi
    chars = " .:-=+*#%@"
    nch = len(chars)
    out = []
    for r in range(nrow):
        line = ""
        for c in range(ncol):
            val = g[r, c]
            if transform == 'presence':
                line += '#' if val > 0 else '.'
            elif transform == 'parity':
                line += '#' if (int(round(val)) % 2 == 1) else '.'
            elif transform == 'value':
                idx = min(nch - 1, int(round(val)))
                line += chars[idx] if val > 0 else ' '
            elif transform == 'mod2':
                line += '#' if (val > 0 and int(round(val)) % 2 == 1) else ('.' if val > 0 else ' ')
        out.append(line)
    return out, g


def show(lines):
    for l in lines:
        print(l)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("counts", help="counts.csv path")
    ap.add_argument("--cell", type=float, default=60)
    ap.add_argument("--transform", default="value", choices=["value", "presence", "parity", "mod2"])
    args = ap.parse_args()

    counts = load_counts(args.counts)
    plots = pd.read_csv(os.path.join(DATA, "survey_plots.csv"))
    df = plots.merge(counts, on="image", how="left")
    df["count"] = df["count"].fillna(0)
    lines, g = render_grid(df, "x_m", "y_m", "count", args.cell, args.transform)
    print(f"=== transform={args.transform} cell={args.cell} grid {len(lines)}x{len(lines[0])} ===")
    show(lines)