#!/usr/bin/env python3
"""Reproducible extraction of the earned chart from the participant engine.

The native engine embeds the qualifying chart as a static JSON literal in
.rodata (the same literal is parsed and attached to `eval-finish` as
`summary["chart"]` once the 512-game evaluation qualifies).  This script:

  1. locates and parses that literal from the supplied `samoa-linux-x86_64`;
  2. reconstructs the five "lights" (size > 6) and their companions
     (size <= 6), with companions ordered nearest -> farthest;
  3. prints the natural readings and the candidate decodings explored.

Usage:
    /home/kozak/mllm/py decode_chart.py [path-to-samoa-linux-x86_64]
"""
import json
import math
import sys
from pathlib import Path

DEFAULT_BIN = (
    Path(__file__).resolve().parents[2]
    / "tasks/task_samoa/bundle/extracted/island-duel-participant/samoa-linux-x86_64"
)


def load_chart(path: Path):
    raw = path.read_bytes()
    start = raw.index(b'{"prefix":"SIGMOID_"')
    text = raw[start:start + 4000].decode("utf-8", "ignore")
    chart, consumed = json.JSONDecoder().raw_decode(text)
    return chart, start, consumed


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_BIN
    chart, offset, consumed = load_chart(path)
    print(f"chart literal at file offset 0x{offset:x} ({consumed} bytes)")
    print("title:", chart["title"])
    for line in chart["verse"]:
        print("verse:", line)
    stars = chart["stars"]
    guides = [s for s in stars if s["size"] > 6]
    comps = [s for s in stars if s["size"] <= 6]

    def dist(a, b):
        return math.hypot(a["x"] - b["x"], a["y"] - b["y"])

    groups = {}
    for g in guides:
        mine = [c for c in comps if min(dist(h, c) for h in guides) == dist(g, c)]
        mine.sort(key=lambda c: dist(g, c))
        groups[g["text"]] = [(round(dist(g, c), 3), c["text"]) for c in mine]
        print(f"guide {g['text']} at ({g['x']},{g['y']}) size {g['size']}:", groups[g['text']])

    # Flagship reading: Southern-Cross Greek order (T,N,E,C,I) / near->far.
    order = "TNECI"
    seq = "".join(t for g in order for _, t in groups[g])
    print("\nreading (Greek order T,N,E,C,I; near->far):", seq)

    def caesar(s, k):
        return "".join(chr((ord(c) - 65 + k) % 26 + 65) for c in s)

    def progressive(s, k=1, start=1):
        return "".join(
            chr((ord(c) - 65 - (k * i + start)) % 26 + 65) for i, c in enumerate(s)
        )

    print("plain           :", seq)
    print("undo +1,+2,...  :", progressive(seq))
    print("undo +0,+1,...  :", progressive(seq, 1, 0))
    print("undo +k=5       :", progressive(seq, 5, 1))
    print("ROT13           :", caesar(seq, 13))
    print("Atbash          :", "".join(chr(90 - (ord(c) - 65)) for c in seq))


if __name__ == "__main__":
    main()
