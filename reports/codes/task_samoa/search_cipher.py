"""Search chart readings x cipher families for an English 12-letter word."""
import itertools
import json
import math
from pathlib import Path

BIN = Path(__file__).resolve().parents[2] / "tasks/task_samoa/bundle/extracted/island-duel-participant/samoa-linux-x86_64"
raw = BIN.read_bytes()
st = raw.index(b'{"prefix":"SIGMOID_"')
chart, _ = json.JSONDecoder().raw_decode(raw[st:st + 4000].decode("utf-8", "ignore"))
stars = chart["stars"]
G = [s for s in stars if s["size"] > 6]
C = [s for s in stars if s["size"] <= 6]
D = lambda a, b: math.hypot(a["x"] - b["x"], a["y"] - b["y"])

groups = {}
for g in G:
    mine = [c for c in C if min(D(h, c) for h in G) == D(g, c)]
    groups[g["text"]] = [c["text"] for c in sorted(mine, key=lambda c: D(g, c))]

VOW = set("AEIOU")
TRIGRAMS = ["THE", "AND", "ING", "ION", "ENT", "ATI", "TIO", "HER", "FOR", "THA",
            "NAV", "VOY", "STA", "HOM", "CRO", "WAY", "POL", "SAI", "SEA", "CHA",
            "LIG", "NIG", "SKY", "OCE", "TID", "WIN", "ISL", "PAC", "SAM", "CAN",
            "MAS", "COM", "GUI", "STE", "PRO", "TAR", "ART", "OUR", "ONE", "OUT",
            "EAR", "HOR", "ORT", "RTH", "SOU", "UTH", "ERN", "RSE", "OSS", "FIN",
            "IND", "DIS", "COV", "OVE", "VER", "EVE", "ERY", "RYT", "YTH", "THI"]
THEME = ["NAVIG", "VOYAG", "STAR", "HOMEW", "CROSS", "WAYFI", "POLYN", "SAIL",
         "OCEAN", "PACIF", "SAMOA", "ISLAN", "CANOE", "COMPA", "CHART", "LIGHT",
         "NIGHT", "SOUTH", "NORTH", "HORIZ", "GUIDE", "PASSAG", "TREAS", "FOUND"]

def score(s):
    sc = sum(3 for t in TRIGRAMS if t in s)
    sc += sum(10 for t in THEME if t in s)
    # vowel/consonant alternation bonus
    alt = sum(1 for i in range(len(s) - 1) if (s[i] in VOW) != (s[i + 1] in VOW))
    sc += alt * 0.5
    sc += sum(0.3 for ch in s if ch in "ETAOINSHRDLU")
    return sc

def shift(s, k):
    return "".join(chr((ord(c) - 65 + k) % 26 + 65) for c in s)

def vig_dec(s, key):
    return "".join(chr((ord(c) - 65 - (ord(key[i % len(key)]) - 65)) % 26 + 65) for i, c in enumerate(s))

def atbash(s):
    return "".join(chr(90 - (ord(c) - 65)) for c in s)

names = [g["text"] for g in G]
results = []
# ciphertext orders: guide-order permutations, intra near->far (and reversed)
for order in itertools.permutations(names):
    for rev in (False, True):
        seq = "".join("".join(groups[g][::-1] if rev else groups[g]) for g in order)
        key = "".join(order)
        cands = {}
        for k in range(26):
            cands[f"caesar{k}"] = shift(seq, k)
        for k in range(26):
            cands[f"atbash{k}"] = shift(atbash(seq), k)
        for k in range(1, 26):
            cands[f"prog{k}"] = "".join(chr((ord(c) - 65 - k * i) % 26 + 65) for i, c in enumerate(seq))
            cands[f"progB{k}"] = "".join(chr((ord(c) - 65 - k * (i + 1)) % 26 + 65) for i, c in enumerate(seq))
            cands[f"progAdd{k}"] = "".join(chr((ord(c) - 65 + k * i) % 26 + 65) for i, c in enumerate(seq))
        for keyname, kk in [("guide", key), ("revguide", key[::-1])]:
            cands["vig_" + keyname] = vig_dec(seq, kk)
            cands["vigdec2_" + keyname] = vig_dec(seq, kk[::-1])
        for nm, out in cands.items():
            results.append((score(out), out, seq, order, rev, nm))

results.sort(reverse=True)
seen = set()
for sc, out, seq, order, rev, nm in results[:400]:
    if out in seen:
        continue
    seen.add(out)
    if sc >= 0:
        print(f"{sc:5.1f} {out}  <= seq={seq} order={order} rev={rev} {nm}")
    if len(seen) > 50:
        break
