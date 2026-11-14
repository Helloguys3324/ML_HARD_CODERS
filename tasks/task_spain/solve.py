#!/usr/bin/env python3
"""task_spain  --  "¡Feliz Navidad!"  (LMML 2026, team HardCoders)

Rebuild the office Christmas board:

  1. unit price  = Total_paid_eur / Quantity           (Quantity is in the
     office-guide unit: kg / dozen / bar / kit / g / box)
  2. office rule : a souvenir counts iff  Fair_min_eur <= unit_price <= Fair_max_eur
     (cheaper = fake, dearer = tourist trap -> neither is checked)
  3. board       : rows = the 6 souvenirs in List_no order,
                   columns = the 21 pilots in Pilot_no order,
                   cell = 1 iff that pilot bought that souvenir at a fair price
                   at least once ("HR checks which pilot brought a given
                   souvenir for a fair price").
  4. The 6x21 grid is a pixel image.  Read with 3-wide glyphs separated by
     blank columns -> the board spells the flag word.

Usage:
    /home/kozak/mllm/py solve.py [DATA_DIR] [OUT_DIR]
Defaults: DATA_DIR=../../tasks/task_spain/data (relative to this file),
          OUT_DIR=this directory.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.normpath(
    os.path.join(HERE, "..", "..", "tasks", "task_spain", "data"))
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else HERE
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 1. load
# ----------------------------------------------------------------------
guide = pd.read_csv(os.path.join(DATA_DIR, "office_price_guide.csv"))
purch = pd.read_csv(os.path.join(DATA_DIR, "purchases.csv"))
guide = guide.sort_values("List_no").reset_index(drop=True)

# sanity checks on the schema
assert list(guide["List_no"]) == [1, 2, 3, 4, 5, 6]
assert set(guide["Product"]) == set(purch["Product"])

purch["unit_price"] = purch["Total_paid_eur"] / purch["Quantity"]
df = purch.merge(
    guide[["List_no", "Product", "Unit", "Fair_min_eur", "Fair_max_eur", "Home_city"]],
    on="Product", how="left", validate="many_to_one")

df["verdict"] = np.where(df["unit_price"] < df["Fair_min_eur"], "fake",
                  np.where(df["unit_price"] > df["Fair_max_eur"], "trap", "fair"))
df["fair"] = df["verdict"] == "fair"

print("receipts:", len(df))
print(df["verdict"].value_counts().to_string())
print("city matches home city:",
      f"{(df['City'] == df['Home_city']).mean():.1%}")

# ----------------------------------------------------------------------
# 2. board: rows = products (List_no order), columns = pilots 1..21
# ----------------------------------------------------------------------
products = list(guide["Product"])
pilots = list(range(1, 22))

counts = (df[df["fair"]]
          .pivot_table(index="Product", columns="Pilot_no",
                       values="Receipt_id", aggfunc="count")
          .reindex(index=products, columns=pilots)
          .fillna(0)
          .astype(int))
board = (counts > 0).astype(int)          # HR check = at least one fair purchase

print("\n6x21 office board ('#' = checked):")
print("      " + "".join(str(p % 10) for p in pilots))
for prod, row in board.iterrows():
    print(f"{prod[:9]:9s}" + "".join("#" if v else "." for v in row))

# ----------------------------------------------------------------------
# 3. read the pixel word: glyphs separated by all-zero columns
# ----------------------------------------------------------------------
B = board.values
empty = [j + 1 for j in range(B.shape[1]) if B[:, j].sum() == 0]
print("\nblank columns (letter spacers):", empty)
groups, start = [], 0
for j in range(B.shape[1]):
    if B[:, j].sum() == 0:
        if start < j:
            groups.append((start, j))       # columns start..j-1 (0-based)
        start = j + 1
if start < B.shape[1]:
    groups.append((start, B.shape[1]))
print("glyph column groups (1-based, inclusive):",
      [(a + 1, b) for a, b in groups])

# pixel glyph -> letter, keyed by the exact rows of the bitmap
# (row order = office List_no order, top row first)
GLYPHS = {
    ".../.../###/#../#../###": "C",   # lower-case c
    "##./#.#/#.#/##./#.#/#.#": "R",   # R
    ".../.../#.#/#.#/#.#/###": "U",   # lower-case u
    "###/#../#../#../#../###": "C",   # upper-case C
    ".#.###": "I",                    # 1-column lower-case i (dot + stem)
    "###/#.#/#.#/#.#/#.#/###": "O",   # upper-case O
}

word = []
for a, b in groups:
    g = B[:, a:b]
    if g.shape[1] == 1:
        key = "".join("#" if g[r, 0] else "." for r in range(6))
    else:
        key = "/".join("".join("#" if v else "." for v in g[r])
                       for r in range(6))
    letter = GLYPHS.get(key)
    if letter is None:
        raise SystemExit(f"unrecognised glyph at columns {a + 1}-{b}:\n{key}")
    word.append(letter)

WORD = "".join(word)
print("\nglyph letters:", " ".join(word))
print("BOARD WORD   :", WORD)
print("FLAG         : SIGMOID_" + WORD)

with open(os.path.join(OUT_DIR, "answer.txt"), "w") as f:
    f.write("SIGMOID_" + WORD + "\n")

# ----------------------------------------------------------------------
# 4. figure: the board, clearly readable, plus the receipt classification
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [21, 10]})

ax1.imshow(1 - B, cmap="gray", vmin=0, vmax=1, aspect="equal",
           interpolation="nearest")
ax1.set_xticks(range(21))
ax1.set_xticklabels(pilots, fontsize=7)
ax1.set_yticks(range(6))
ax1.set_yticklabels(products, fontsize=8)
ax1.set_xlabel("Pilot_no (board column)")
ax1.set_title("Office Christmas board: fair-price purchases\n"
              f"board spells: {WORD}", fontsize=11)
ax1.set_xticks(np.arange(-0.5, 21, 1), minor=True)
ax1.set_yticks(np.arange(-0.5, 6, 1), minor=True)
ax1.grid(which="minor", color="red", lw=0.4, alpha=0.6)
ax1.tick_params(which="minor", length=0)

# right panel: per-receipt unit price relative to the fair window
for prod, sub in df.groupby("Product", sort=False):
    rel = sub["unit_price"] / sub["Fair_min_eur"]
    y = products.index(prod)
    colors = sub["verdict"].map({"fake": "tab:red", "fair": "tab:green",
                                 "trap": "tab:orange"})
    ax2.scatter(rel, np.full(len(sub), y) + np.random.default_rng(0).uniform(
        -0.18, 0.18, len(sub)), c=colors, s=7, alpha=0.7)
ax2.set_yticks(range(6))
ax2.set_yticklabels(products, fontsize=8)
ax2.set_xscale("log")
ax2.axvline(1.0, color="k", lw=0.8, ls="--")
ax2.set_xlabel("unit price / Fair_min  (log scale)")
ax2.set_title("Receipts: fake (<1), fair (1..max/min), tourist trap\n"
              "green = counted on the board", fontsize=10)
ax2.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "board_solution.png"), dpi=180)
print("\nwrote", os.path.join(OUT_DIR, "board_solution.png"))
print("wrote", os.path.join(OUT_DIR, "answer.txt"))
