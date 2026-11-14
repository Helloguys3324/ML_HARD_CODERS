#!/usr/bin/env python3
"""Decode the 8-crate counts into an 8-letter word.

Reads counts.csv (crate,set,glyph_id,count) or the JSON dump, tries a battery of
encodings, prints candidate words.
"""
from __future__ import annotations

import argparse
import string

import numpy as np
import pandas as pd

CRATES = ["crate_01", "crate_02", "crate_03", "crate_04",
          "crate_05", "crate_06", "crate_07", "crate_08"]


def a2z(v: int) -> str | None:
    if 1 <= v <= 26:
        return string.ascii_uppercase[v - 1]
    return None


def word_from(vals, base: int = 0):
    out = []
    for v in vals:
        if base:
            v = v % base
        c = a2z(int(round(v)))
        out.append(c if c else "?")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default="challenges/task_china/counts.csv")
    a = ap.parse_args()

    df = pd.read_csv(a.counts)
    # normalize crate/set ordering
    m = {c: [] for c in CRATES}
    for c in CRATES:
        sub = df[df.crate == c].sort_values("set")
        m[c] = sub["count"].astype(int).tolist()

    print("Raw counts per crate per set:")
    for c in CRATES:
        print(f"  {c}: {m[c]}")

    sums = [sum(m[c]) for c in CRATES]
    meds = [int(np.median(m[c])) for c in CRATES]
    mins = [int(np.min(m[c])) for c in CRATES]
    maxs = [int(np.max(m[c])) for c in CRATES]
    sets_equal = all(len(set(m[c])) == 1 for c in CRATES)

    print("\nper-crate sum :", sums)
    print("per-crate med :", meds)
    print("per-crate min :", mins)
    print("per-crate max :", maxs)
    print("all 5 sets identical within each crate?:", sets_equal)

    print("\n-- candidate words --")
    print("sum->A1Z26    :", word_from(sums))
    print("sum%26->A1Z26 :", word_from(sums, 26))
    print("med->A1Z26    :", word_from(meds))
    print("med%26        :", word_from(meds, 26))

    # if all sets equal, the common value is the count
    if sets_equal:
        print("common count->A1Z26:", word_from(meds))
        print("common count%26    :", word_from(meds, 26))

    # per-set column readings (transpose): 5 columns, 8 rows
    mat = np.array([m[c] for c in CRATES])  # 8 x 5
    print("\ncount matrix (8 crates x 5 sets):")
    print(mat.T)  # 5 set-rows x 8 crate-cols
    print("column (set) sums/medians could encode — inspect manually.")


if __name__ == "__main__":
    main()