#!/usr/bin/env python3
"""Decode survey as a 9x37 grid of per-plot sheep counts.

The drone surveyed a raster: 9 rows x 37 columns (50m spacing). Each plot's
count is a grid cell value. Render the grid as ASCII to read the word.
"""
import os
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "data")


def make_grid(counts_path):
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

    # 9 rows by S (sort and split into 9 groups of 37)
    df = df.sort_values("S").reset_index(drop=True)
    assert len(df) == 333, len(df)
    n_rows, n_cols = 9, 37
    grid = np.zeros((n_rows, n_cols))
    for r in range(n_rows):
        chunk = df.iloc[r * n_cols:(r + 1) * n_cols].sort_values("L")
        grid[r, :] = chunk["count"].values
    return grid, df


def show(grid, transform="value"):
    chars = " .:-=+*#%@"
    nch = len(chars)
    n_r, n_c = grid.shape
    for r in range(n_r):
        line = ""
        for cc in range(n_c):
            v = grid[r, cc]
            if transform == "value":
                idx = min(nch - 1, int(round(v)))
                line += chars[idx] if v > 0 else "."
            elif transform == "presence":
                line += "#" if v > 0 else "."
            elif transform == "parity":
                line += "#" if int(round(v)) % 2 == 1 else "."
            elif transform == "thresh":
                med = np.median(grid)
                line += "#" if v > med else "."
        print(line)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("counts")
    ap.add_argument("--transform", default="value",
                    choices=["value", "presence", "parity", "thresh"])
    args = ap.parse_args()
    grid, df = make_grid(args.counts)
    print(f"=== grid {grid.shape} transform={args.transform} (rows top->bottom along +S) ===")
    show(grid, args.transform)
    print("\ncount stats: min %.0f max %.0f mean %.1f" % (grid.min(), grid.max(), grid.mean()))