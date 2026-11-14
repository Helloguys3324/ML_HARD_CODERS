#!/usr/bin/env python3
"""Decode counts_all.csv (multi-threshold) into an 8-letter word.

Tries a battery of encodings per threshold and prints candidate words.
"""
from __future__ import annotations

import argparse
import json
import string
from pathlib import Path

import numpy as np
import pandas as pd

CRATES = [f"crate_{i:02d}" for i in range(1, 9)]


def A(n):
    n = int(round(n))
    return string.ascii_uppercase[n - 1] if 1 <= n <= 26 else "?"


def words_from_matrix(M: np.ndarray, tag: str):
    """M is 8 x 5 (crate x set) integer counts."""
    print(f"[{tag}]")
    for crate, row in zip(CRATES, M):
        print(f"   {crate}: {row.tolist()}")
    sums = M.sum(1)
    meds = np.median(M, 1)
    mins = M.min(1)
    print("   sums:", sums.tolist())
    print("   word(sum A1Z26):", "".join(A(s) for s in sums))
    print("   word(sum%26 A1Z26):", "".join(A(s % 26) if (s % 26) != 0 else 'Z' for s in sums))
    # if uniform within crate
    uniform = [len(set(r.tolist())) == 1 for r in M]
    if all(uniform):
        vals = M[:, 0].astype(int)
        print("   ALL 5 sets uniform per crate! common:", vals.tolist())
        print("   word(common A1Z26):", "".join(A(v) for v in vals))
        print("   word(common%26):", "".join(A(v % 26) if (v % 26) != 0 else 'Z' for v in vals))
    # column sums (per set across crates)
    colsums = M.sum(0)
    print("   per-set sums (across crates):", colsums.tolist())
    # differences set-to-set
    print("   min/max per crate:", [(int(r.min()), int(r.max())) for r in M])
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="challenges/task_china/kaggle/out/counts_all.csv")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    for thr in sorted(df.threshold.unique()):
        sub = df[df.threshold == thr]
        M = np.zeros((8, 5), dtype=int)
        for i, crate in enumerate(CRATES):
            r = sub[sub.crate == crate].sort_values("set")
            M[i] = r["count"].values
        words_from_matrix(M, f"threshold={thr}")


if __name__ == "__main__":
    main()