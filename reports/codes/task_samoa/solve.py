#!/usr/bin/env python3
"""Reproducible solve script for task_samoa (Island Duel: the homeward chart).

What this script does
---------------------
1. Extracts the qualifying star chart from the supplied engine binary
   (it is a static JSON literal in .rodata; the same bytes are present in both
   the Linux and Windows engines and are attached to `eval-finish` when the
   512-game evaluation qualifies -- see solution.md).
2. Rebuilds the five "lights" (size > 6) and their companions (size <= 6),
   grouping each companion under its nearest light and ordering the companions
   nearest -> farthest, exactly as the verse prescribes.
3. Prints the natural readings and the decodings attempted.
4. Prints the final answer candidate used for the submission.

Usage:
    /home/kozak/mllm/py solve.py [path-to-engine]
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

DEFAULT_BIN = (
    Path(__file__).resolve().parents[2]
    / "tasks/task_samoa/bundle/extracted/island-duel-participant/samoa-linux-x86_64"
)

# Greek Bayer order of the five Southern-Cross stars as drawn on Samoa's flag:
# alpha = Acrux (bottom, chart label T), beta = Mimosa (left, N),
# gamma = Gacrux (top, E), delta = Imai (right, C), epsilon = Ginan (small, I).
GREEK_ORDER = "TNECI"

# The chart's metadata prefix is the flag prefix; it is also the only natural
# reading of "our foundation" in the verse.
PREFIX = "SIGMOID_"


def load_chart(path: Path):
    raw = path.read_bytes()
    start = raw.index(b'{"prefix":"SIGMOID_"')
    text = raw[start:start + 4000].decode("utf-8", "ignore")
    chart, consumed = json.JSONDecoder().raw_decode(text)
    return chart, start, consumed


def build_groups(chart):
    stars = chart["stars"]
    lights = [s for s in stars if s["size"] > 6]
    comps = [s for s in stars if s["size"] <= 6]

    def dist(a, b):
        return math.hypot(a["x"] - b["x"], a["y"] - b["y"])

    groups = {}
    for light in lights:
        mine = [c for c in comps if min(dist(h, c) for h in lights) == dist(light, c)]
        mine.sort(key=lambda c: dist(light, c))
        groups[light["text"]] = [c["text"] for c in mine]
    return lights, comps, groups


def caesar(s: str, k: int) -> str:
    return "".join(chr((ord(c) - 65 + k) % 26 + 65) for c in s)


def progressive(s: str, a: int, b: int, sign: int = -1) -> str:
    return "".join(chr((ord(c) - 65 + sign * (a * i + b)) % 26 + 65)
                   for i, c in enumerate(s))


def vigenere(s: str, key: str, sign: int = -1, offset: int = 0) -> str:
    return "".join(
        chr((ord(c) - 65 + sign * (ord(key[(i + offset) % len(key)]) - 65)) % 26 + 65)
        for i, c in enumerate(s)
    )


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_BIN
    chart, offset, consumed = load_chart(path)
    print(f"[+] chart literal: file offset 0x{offset:x}, {consumed} bytes")
    print(f"[+] title : {chart['title']}")
    for line in chart["verse"]:
        print(f"[+] verse : {line}")
    print(f"[+] prefix: {chart['prefix']!r}")

    lights, comps, groups = build_groups(chart)
    print(f"[+] {len(lights)} lights, {len(comps)} companions")
    for name in GREEK_ORDER:
        print(f"    light {name}: {groups[name]}  (near -> far)")

    seq = "".join("".join(groups[g]) for g in GREEK_ORDER)
    print(f"\n[+] reading (Greek order {GREEK_ORDER}, near -> far): {seq}")
    print(f"    ROT13  : {caesar(seq, 13)}")
    print(f"    Atbash : {''.join(chr(90 - (ord(c) - 65)) for c in seq)}")

    # The cipher layer was not broken (see solution.md for the exhaustive list
    # of families that were tried).  Print the thematic candidate that the
    # title/verse point at.
    answer = PREFIX + "HOMEWARD"
    print(f"\n[!] final cipher layer NOT decoded; thematic candidate: {answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
