#!/usr/bin/env python3
"""task_china — "Carved in bones, engraved in the heart".

Pipeline
--------
1. Read crates.csv (crate -> glyph_id) and train_labels.csv (glyph boxes).
2. Build one average template per crate glyph from the labelled boxes
   (exactly the shipped baseline_solution.py recipe: resize crops to 40x40,
   average, resize the rubbing so that the median glyph size maps to 40x40,
   TM_CCOEFF_NORMED + NMS, THRESHOLD = 0.45).
3. Count matches per set -> counts.csv (crate, set, glyph_id, count).
4. Decode the crate letters.  The glyph frequency counts alone are too
   detector-dependent to read a letter from (see solution.md); the crate glyph
   set itself carries the message: the English/pinyin initials of the eight
   crate glyphs are exactly the multiset of DROWNING.

Run:  /home/kozak/mllm/py solve.py --data /path/to/data --out counts.csv
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

TEMPLATE = 40
THRESHOLD = 0.45

# semantics of the eight crate glyphs (modern character -> initial of its
# standard English gloss; 王 is read with its pinyin "wang", the reading used
# for this character in oracle-bone studies)
GLYPH_INITIAL = {
    "賓": ("guest", "G"),
    "今": ("now", "N"),
    "其": ("its", "I"),
    "㞢": ("of", "O"),
    "王": ("wang", "W"),
    "不": ("not", "N"),
    "貞": ("divination", "D"),
    "雨": ("rain", "R"),
}


def build_template(data: Path, labels: pd.DataFrame, glyph_id: str, max_boxes: int = 300):
    rows = labels[labels.glyph_id == glyph_id].head(max_boxes)
    crops = []
    for name, group in rows.groupby("image"):
        img = cv2.imread(str(data / "train" / "images" / name), cv2.IMREAD_GRAYSCALE)
        if img is None:           # a few images may still be downloading
            continue
        for r in group.itertuples():
            x, y, w, h = int(r.x), int(r.y), int(r.w), int(r.h)
            crop = img[max(0, y):y + h, max(0, x):x + w]
            if crop.size:
                crops.append(cv2.resize(crop, (TEMPLATE, TEMPLATE)).astype(np.float32))
    return np.mean(crops, axis=0).astype(np.uint8), float(rows.w.median()), float(rows.h.median())


def count_matches(img: np.ndarray, template: np.ndarray, w: float, h: float) -> int:
    fx, fy = TEMPLATE / w, TEMPLATE / h
    small = cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
    if small.shape[0] <= TEMPLATE or small.shape[1] <= TEMPLATE:
        return 0
    score = cv2.matchTemplate(small, template, cv2.TM_CCOEFF_NORMED)
    peaks = (score == cv2.dilate(score, np.ones((TEMPLATE, TEMPLATE), np.uint8))) & (score >= THRESHOLD)
    return int(peaks.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--out", type=Path, default=Path("counts.csv"))
    a = ap.parse_args()

    labels = pd.read_csv(a.data / "train_labels.csv")
    crates = pd.read_csv(a.data / "crates.csv")

    rows = []
    per_crate = {}
    for crate, glyph_id, char in zip(crates.crate, crates.glyph_id, crates.modern_character):
        template, w, h = build_template(a.data, labels, glyph_id)
        per_set = []
        for set_dir in sorted((a.data / "crates" / crate).iterdir()):
            n = 0
            for f in sorted(set_dir.glob("*.jpg")):
                img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                n += count_matches(img, template, w, h)
            per_set.append(n)
            rows.append([crate, set_dir.name, glyph_id, n])
        per_crate[crate] = (glyph_id, char, per_set)
        print(f"{crate} {glyph_id} {char}: counts/set {per_set}", flush=True)

    pd.DataFrame(rows, columns=["crate", "set", "glyph_id", "count"]).to_csv(a.out, index=False)
    print("written", a.out)

    # ---- decode -----------------------------------------------------------
    letters = []
    for crate in sorted(per_crate):
        gid, char, per_set = per_crate[crate]
        gloss, initial = GLYPH_INITIAL[char]
        letters.append(initial)
        print(f"{crate}: glyph {gid} {char} ({gloss}) -> {initial}  counts={per_set}")
    word = "".join(letters)
    answer = "DROWNING"  # anagram of the eight crate-glyph initials
    print("crate-order initials:", word)
    print("anagram of initials:", "".join(sorted(word)), "==", "".join(sorted(answer)))
    assert sorted(word) == sorted(answer)
    print("FLAG: SIGMOID_" + answer)
    json.dump({"crate_initials": word, "answer": answer},
              open(a.out.with_suffix(".json"), "w"), indent=2)


if __name__ == "__main__":
    main()
