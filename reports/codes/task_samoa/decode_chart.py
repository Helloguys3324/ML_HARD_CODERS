"""Decode the Island Duel star chart -> flag word.

The chart JSON is embedded in the engine binary (same for every qualifying run).
Here we recover the reading order + cipher/rotation described by the verse:
  1. "The five lights of home keep the Greek watch." -> order guides by Greek
     letters (a,b,g,d,e) of the Southern Cross.
  2. "Each keeps its nearer companions before the farther." -> companions sorted
     by distance from their guide.
  3. "Our foundation turns the alphabet, over and over." -> a rotation cipher.
  4. "For the homeward message, undo those turns." -> reverse the rotation.
"""
import json
import math
import itertools

CHART = {
  "prefix": "SIGMOID_",
  "title": "The navigator's homeward chart",
  "verse": [
    "The five lights of home keep the Greek watch.",
    "Each keeps its nearer companions before the farther.",
    "Our foundation turns the alphabet, over and over.",
    "For the homeward message, undo those turns.",
  ],
  "stars": [
    {"x": 40,        "y": 88,       "size": 10,  "text": "N"},
    {"x": 97.454,    "y": 197.819,  "size": 3.5, "text": "C"},
    {"x": 96,        "y": 28,       "size": 10,  "text": "E"},
    {"x": 110.092,   "y": 137.003,  "size": 3.5, "text": "U"},
    {"x": 125.985,   "y": 169.57,   "size": 3.5, "text": "V"},
    {"x": 100,       "y": 180,      "size": 10,  "text": "T"},
    {"x": 160,       "y": 76,       "size": 10,  "text": "C"},
    {"x": 22.828,    "y": 82.603,   "size": 3.5, "text": "M"},
    {"x": 112.355,   "y": 20.482,   "size": 3.5, "text": "Y"},
    {"x": 46.086,    "y": 115.331,  "size": 3.5, "text": "H"},
    {"x": 56.943,    "y": 72.446,   "size": 3.5, "text": "K"},
    {"x": 176.997,   "y": 70.076,   "size": 3.5, "text": "N"},
    {"x": 145.319,   "y": 142.108,  "size": 3.5, "text": "Z"},
    {"x": 81.908,    "y": 165.799,  "size": 3.5, "text": "E"},
    {"x": 155.696,   "y": 98.594,   "size": 3.5, "text": "V"},
    {"x": 93.871,    "y": 50.901,   "size": 3.5, "text": "Y"},
    {"x": 132,       "y": 130,      "size": 7,   "text": "I"},
  ],
  "view_box": [0, -15, 220, 250],
}

STARS = CHART["stars"]
GUIDES = [s for s in STARS if s["size"] > 6]
COMPS = [s for s in STARS if s["size"] <= 6]
GUIDES.sort(key=lambda s: -s["size"])


def d(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def assign(group_guide_order, near_first=True):
    """group companions by nearest guide; return list of (guide_text, [comp_texts])"""
    res = []
    for gtxt in group_guide_order:
        g = next(s for s in GUIDES if s["text"] == gtxt)
        cs = [c for c in COMPS if min(GUIDES, key=lambda gg: d(gg, c))["text"] == gtxt]
        cs.sort(key=lambda c: d(g, c), reverse=not near_first)
        res.append((gtxt, [c["text"] for c in cs]))
    return res


def make_strings(order, near_first, include_guides, guide_first):
    groups = assign(order, near_first)
    parts = []
    for gtxt, cts in groups:
        items = cts + ([gtxt] if include_guides else [])
        if guide_first and include_guides:
            items = [gtxt] + cts
        parts.extend(items)
    return "".join(parts)


def caesar(s, k):
    return "".join(chr((ord(c) - 65 - k) % 26 + 65) for c in s)


def vigenere_dec(s, key):
    out = []
    ki = 0
    for c in s:
        k = ord(key[ki % len(key)]) - 65
        out.append(chr((ord(c) - 65 - k) % 26 + 65))
        ki += 1
    return "".join(out)


def atbash(s):
    return "".join(chr(25 - (ord(c) - 65) + 65) for c in s)


# English scoring via common bigrams/trigrams (rough)
COMMON = set("THE AND ING HER ERE ENT THA NTH WAS ETH FOR DTH HAT SHE ION INT HIS THI YOU VER TER ATE ATI ALL OUR OUL EST TED AIT ERE REA RES REC ANC".split())
BIGRAMS = "th he in er an re on in en nd ti te at on nt ha es st en ed of to it al ar is as de".split()


def score(s):
    ls = s.lower()
    sc = 0.0
    for tri in COMMON:
        if tri in ls:
            sc += len(tri)
    for bg in BIGRAMS:
        sc += ls.count(bg) * 0.5
    # vowel ratio sanity
    v = sum(1 for c in ls if c in "aeiou")
    if len(s) > 0 and 0.2 < v / len(s) < 0.7:
        sc += 2
    return sc


if __name__ == "__main__":
    orders = {
        "greek_TNECI": ["T", "N", "E", "C", "I"],
        "greek_TCENI": ["T", "C", "E", "N", "I"],
        "greek_TNEIC": ["T", "N", "E", "I", "C"],
        "bright": ["T", "N", "E", "C", "I"],
        "byY": [s["text"] for s in sorted(GUIDES, key=lambda s: s["y"])],
        "byX": [s["text"] for s in sorted(GUIDES, key=lambda s: s["x"])],
    }
    # angular around center I
    cx = 132.0; cy = 130.0
    import math as _m
    ang = sorted(GUIDES, key=lambda s: _m.atan2(s["y"] - cy, s["x"] - cx))
    orders["ang_ccw"] = [s["text"] for s in ang]
    orders["ang_cw"] = [s["text"] for s in ang[::-1]]

    keys = ["CENTI", "TNECI", "CRUX", "SAMOA", "APIA", "SIGMOID", "FOUNDATION",
            "FIVE", "HOME", "BASIS", "BASE", "GREEK", "ALPHA", "WATCH", "SOUTH",
            "CROSS", "STARS", "OCEAN", "CANOE", "VOYAGE", "HOMEWARD"]

    results = []
    for oname, order in orders.items():
        for near_first in (True, False):
            for inc_g in (True, False):
                for gf in (True, False):
                    if not inc_g and gf:
                        continue
                    s = make_strings(order, near_first, inc_g, gf)
                    # caesar
                    for k in range(26):
                        results.append((score(caesar(s, k)), f"caesar{k}", oname, near_first, inc_g, gf, caesar(s, k)))
                    # vigenere
                    for key in keys:
                        results.append((score(vigenere_dec(s, key)), f"vig {key}", oname, near_first, inc_g, gf, vigenere_dec(s, key)))
    results.sort(reverse=True, key=lambda r: r[0])
    seen = set()
    for r in results[:80]:
        if r[6] in seen:
            continue
        seen.add(r[6])
        if r[0] > 6:
            print(f"{r[0]:6.1f} {r[1]:14s} {r[2]:12s} nf={r[3]} ig={r[4]} gf={r[5]} => {r[6]}")