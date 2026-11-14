"""Brute-force readings of the chart cipher (geometry + verse)."""
import itertools
import json
import math
from pathlib import Path

BIN = Path(__file__).resolve().parents[2] / "tasks/task_samoa/bundle/extracted/island-duel-participant/samoa-linux-x86_64"
raw = BIN.read_bytes()
start = raw.index(b'{"prefix":"SIGMOID_"')
chart, _ = json.JSONDecoder().raw_decode(raw[start:start + 4000].decode("utf-8", "ignore"))
stars = chart["stars"]
guides = [s for s in stars if s["size"] > 6]
comps = [s for s in stars if s["size"] <= 6]

def dist(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])

groups = {}
for g in guides:
    near = sorted(
        ((dist(g, c), c["text"]) for c in comps if min(dist(h, c) for h in guides) == dist(g, c)),
    )
    groups[g["text"]] = [t for _, t in near]
print("groups nearest:", groups)

# alternative fixed rings: companions within 30
rings = {}
for g in guides:
    near = sorted((dist(g, c), c["text"]) for c in comps if dist(g, c) <= 30)
    rings[g["text"]] = [t for _, t in near]
    print(g["text"], [(round(d, 3), t) for d, t in near])

def uncaesar(s, k, start_i=0):
    return "".join(chr((ord(ch) - 65 - k * (i + start_i)) % 26 + 65) for i, ch in enumerate(s))

def score(s):
    # simple English-likeness: common bigrams/trigrams
    good = ["TH", "HE", "IN", "ER", "AN", "RE", "ON", "AT", "EN", "ND", "TI", "ES",
            "OR", "TE", "OF", "ED", "IS", "IT", "AL", "AR", "ST", "TO", "NT", "NG",
            "SE", "HA", "AS", "OU", "IO", "LE", "VE", "CO", "ME", "DE", "HI", "RI",
            "RO", "IC", "NE", "EA", "RA", "CE", "LI", "CH", "LL", "BE", "MA", "SI",
            "OM", "UR"]
    return sum(2 for t in good if t in s) + sum(1 for v in "AEIOU" if v in s)

for name, gd in [("nearest", groups), ("rings<30", rings)]:
    print("\n===", name)
    for order in itertools.permutations([g["text"] for g in guides]):
        seq = "".join("".join(gd[g]) for g in order)
        for k in range(1, 26):
            for st in (0, 1):
                d = uncaesar(seq, k, st)
                if score(d) >= 8:
                    print(order, "k=", k, "st=", st, seq, "->", d, "score", score(d))
