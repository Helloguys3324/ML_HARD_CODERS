"""Definitive dictionary search over chart readings and simple ciphers."""
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
rings = {}
for g in G:
    mine = [c for c in C if min(D(h, c) for h in G) == D(g, c)]
    groups[g["text"]] = [c["text"] for c in sorted(mine, key=lambda c: D(g, c))]
    rings[g["text"]] = [(round(D(g, c)), c["text"]) for c in sorted(mine, key=lambda c: D(g, c))]
print("groups:", groups)
print("rings:", rings)

WORDS = set(Path("/tmp/en_words.txt").read_text().split())
print("dict size", len(WORDS))
names = [g["text"] for g in G]

def shift(s, k):
    return "".join(chr((ord(c) - 65 + k) % 26 + 65) for c in s)

def prog(s, k, c, sign=-1):
    return "".join(chr((ord(ch) - 65 + sign * (k * i + c)) % 26 + 65) for i, ch in enumerate(s))

def atbash(s):
    return "".join(chr(90 - (ord(c) - 65)) for c in s)

def vig(s, key, sign=-1):
    return "".join(chr((ord(c) - 65 + sign * (ord(key[i % len(key)]) - 65)) % 26 + 65) for i, c in enumerate(s))

keys = ["TNECI", "NETCI", "NETIC", "TCENI", "ACRUXMIMOSAGACRUXIMAI",
        "ALPHABETAGAMMADELTAEPSILON", "ISLANDDUELVOYAGECIPHER", "VOYAGECIPHER",
        "SAMOA", "APIA", "NAVIGATOR", "WAYFINDER", "HOMEWARD", "SOUTHERNCROSS",
        "POLYNESIAN", "STARCHAR", "STAR", "CHART", "LIGHTS", "GREEK", "FOUNDATION",
        "ALPHABET", "SIGMOID", "VOYAGE", "ISLANDDUEL", "CRUX", "ACRUX", "MIMOSA",
        "GACRUX", "IMAI", "GINAN", "EPSILON", "DELTA", "GAMMA", "BETA", "ALPHA",
        "HOMEWARDC", "HOMEWARDCHART", "THEFIVELIGHTS", "OURFOUNDATION"]

found = []

def check(tag, s):
    if s in WORDS or s.lower() in WORDS:
        found.append((tag, s))

# sequences
seqs = {}
for order in itertools.permutations(names):
    for rev in (False, True):
        seq = "".join("".join(groups[g][::-1] if rev else groups[g]) for g in order)
        seqs[f"grp:{''.join(order)}:{rev}"] = seq

# ring-major variants
for order in itertools.permutations(names):
    for rev in (False, True):
        gd = {g: (groups[g][::-1] if rev else groups[g]) for g in order}
        byring = []
        for r in range(3):
            for g in order:
                if len(gd[g]) > r:
                    byring.append(gd[g][r])
        seqs[f"ring:{''.join(order)}:{rev}"] = "".join(byring)

print("sequences", len(seqs))
for tag, seq in seqs.items():
    check(tag + ":plain", seq)
    for k in range(26):
        check(tag + f":caesar{k}", shift(seq, k))
        check(tag + f":atbash{k}", shift(atbash(seq), k))
    for k in range(1, 26):
        for c in range(0, 26):
            check(tag + f":prog{k},{c}", prog(seq, k, c))
            check(tag + f":progadd{k},{c}", prog(seq, k, c, +1))
    for key in keys:
        check(tag + f":vig-{key}", vig(seq, key))
        check(tag + f":vig+{key}", vig(seq, key, +1))
        check(tag + f":beau-{key}", vig(seq, key, +1)[::-1])

print("FOUND:", len(found))
for tag, s in found[:200]:
    print(s, "<=", tag)
