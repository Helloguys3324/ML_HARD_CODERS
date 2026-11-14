"""Broad chart-cipher attack with word-break scoring."""
import itertools
import json
import math
from functools import lru_cache
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
names = [g["text"] for g in G]
WORDS = set(Path("/tmp/en_words.txt").read_text().split())
COMMON = {"THE", "AND", "FOR", "YOU", "ARE", "BUT", "NOT", "ALL", "CAN", "HER",
          "WAS", "ONE", "OUR", "OUT", "DAY", "GET", "HAS", "HIM", "HIS", "HOW",
          "MAN", "NEW", "NOW", "OLD", "SEE", "TWO", "WAY", "WHO", "BOY", "DID",
          "ITS", "LET", "PUT", "SAY", "SHE", "TOO", "USE", "MAY", "OWN", "THE",
          "HOME", "STAR", "SAIL", "FIND", "CROSS", "LIGHT", "NIGHT", "OCEAN",
          "WIND", "WAVE", "CHART", "VOYAGE", "HOMEWARD", "STARS", "NORTH",
          "SOUTH", "EAST", "WEST", "GUIDE", "HELM", "BOAT", "SHIP", "TIDE",
          "SKY", "SEA", "ISLE", "ISLAND", "SAMOA", "APIA", "PACIFIC", "CANOE",
          "MAST", "HULL", "SAILOR", "NAVIGATOR", "WAYFINDER", "POLYNESIAN"}
VOW = set("AEIOU")

def shift(s, k):
    return "".join(chr((ord(c) - 65 + k) % 26 + 65) for c in s)

def prog(s, k, c, sign=-1):
    return "".join(chr((ord(ch) - 65 + sign * (k * i + c)) % 26 + 65) for i, ch in enumerate(s))

def atbash(s):
    return "".join(chr(90 - (ord(c) - 65)) for c in s)

def vig(s, key, sign=-1):
    return "".join(chr((ord(c) - 65 + sign * (ord(key[i % len(key)]) - 65)) % 26 + 65) for i, c in enumerate(s))

def segment_score(s):
    n = len(s)
    dp = [(-1, "")] * (n + 1)
    dp[0] = (0, "")
    for i in range(n):
        if dp[i][0] < 0:
            continue
        for j in range(i + 1, n + 1):
            part = s[i:j]
            if part in WORDS or part in COMMON:
                w = len(part) + (2 if len(part) >= 3 else 0)
                if dp[j][0] < dp[i][0] + w:
                    dp[j] = (dp[i][0] + w, dp[i][1] + " " + part)
    return dp[n]

keys = ["TNECI", "NETCI", "NETIC", "TCENI", "ACRUXMIMOSAGACRUXIMAI",
        "ALPHABETAGAMMADELTAEPSILON", "SAMOA", "APIA", "NAVIGATOR",
        "WAYFINDER", "HOMEWARD", "SOUTHERNCROSS", "POLYNESIAN", "STARCHAR",
        "STAR", "CHART", "LIGHTS", "GREEK", "FOUNDATION", "ALPHABET",
        "SIGMOID", "VOYAGE", "ISLANDDUEL", "CRUX", "ACRUX", "MIMOSA",
        "GACRUX", "IMAI", "GINAN", "EPSILON", "DELTA", "GAMMA", "BETA",
        "ALPHA", "HOMEWARDCHART", "THEFIVELIGHTS", "OURFOUNDATION",
        "TAUNUEPSILONSIGMAIOTA", "TAU", "NU", "EPSILON", "SIGMA", "IOTA",
        "TANGOECHOCHARLIEINDIANOVEMBER", "TANGO", "ECHO", "CHARLIE",
        "INDIA", "NOVEMBER", "LIEUTENANT", "WAYFINDING", "STARCOMPASS"]

seqs = {}
for order in itertools.permutations(names):
    for rev in (False, True):
        gd = {g: (groups[g][::-1] if rev else groups[g]) for g in order}
        seqs[f"grp{''.join(order)}{rev}"] = "".join("".join(gd[g]) for g in order)
        byring = []
        for r in range(3):
            for g in order:
                if len(gd[g]) > r:
                    byring.append(gd[g][r])
        seqs[f"ring{''.join(order)}{rev}"] = "".join(byring)

# geometric orders of all 12 companions (not grouped)
geom = {}
for label, keyfn in [
    ("x", lambda c: (c["x"], c["y"])),
    ("y", lambda c: (c["y"], c["x"])),
    ("cx", lambda c: (math.hypot(c["x"] - 110, c["y"] - 117.5), c["x"])),
    ("ang", lambda c: (math.atan2(c["y"] - 117.5, c["x"] - 110) % (2 * math.pi),)),
]:
    geom["all" + label] = "".join(c["text"] for c in sorted(C, key=keyfn))
    geom["all" + label + "R"] = "".join(c["text"] for c in sorted(C, key=keyfn, reverse=True))
seqs.update(geom)

out = []
for tag, seq in seqs.items():
    variants = [("plain", seq)]
    for k in range(26):
        variants.append((f"cae{k}", shift(seq, k)))
        variants.append((f"atb{k}", shift(atbash(seq), k)))
    for k in range(1, 26):
        for c in range(26):
            variants.append((f"prg{k},{c}", prog(seq, k, c)))
    for key in keys:
        variants.append((f"vig-{key}", vig(seq, key)))
        variants.append((f"vig+{key}", vig(seq, key, 1)))
    for nm, s in variants:
        sc, seg = segment_score(s)
        if sc >= 10:
            out.append((sc, s, seg, tag, nm))

out.sort(reverse=True)
print("candidates", len(out))
seen = set()
for sc, s, seg, tag, nm in out[:200]:
    if s in seen:
        continue
    seen.add(s)
    print(sc, s, "|", seg.strip(), "|", tag, nm)
    if len(seen) >= 40:
        break
