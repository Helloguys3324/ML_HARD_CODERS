"""Exhaustive decode of the star chart using a real English wordlist."""
import json
import math
import itertools

from decode_chart import CHART, GUIDES, COMPS, d, assign

# load wordlist
WORDS = set()
with open("/home/kozak/nltk_data/corpora/words/en") as f:
    for line in f:
        w = line.strip().lower()
        if w.isalpha() and len(w) >= 2:
            WORDS.add(w)

GUIDES.sort(key=lambda s: -s["size"])


def make_strings(order, near_first, include_guides, guide_first):
    groups = assign(order, near_first)
    parts = []
    for gtxt, cts in groups:
        if include_guides:
            items = [gtxt] + cts if guide_first else cts + [gtxt]
        else:
            items = cts
        parts.extend(items)
    return "".join(parts)


def caesar(s, k):
    return "".join(chr((ord(c) - 65 - k) % 26 + 65) for c in s)


def vigenere_dec(s, key):
    out = []
    for i, c in enumerate(s):
        k = ord(key[i % len(key)]) - 65
        out.append(chr((ord(c) - 65 - k) % 26 + 65))
    return "".join(out)


def atbash(s):
    return "".join(chr(25 - (ord(c) - 65) + 65) for c in s)


def is_word(s):
    return s.lower() in WORDS


def splits_words(s):
    """check if s can be split into dictionary words (>=2 letters)."""
    n = len(s)
    def rec(i):
        if i == n:
            return True
        for j in range(i + 1, n + 1):
            if s[i:j].lower() in WORDS and (j - i >= 2):
                if rec(j):
                    return True
        return False
    return rec(0)


matches = []
guide_texts = [g["text"] for g in GUIDES]

# Build all guide permutations (Greek order unknown -> try all; but keep a few plausible first)
from itertools import permutations
perms = list(permutations(guide_texts))
# Also specific orders
keys = ["CENTI", "TNECI", "TCENI", "CRUX", "SAMOA", "APIA", "SIGMOID",
        "FOUNDATION", "FIVE", "HOME", "BASIS", "BASE", "GREEK", "ALPHA",
        "WATCH", "SOUTH", "CROSS", "STARS", "OCEAN", "CANOE", "VOYAGE",
        "HOMEWARD", "NAVIGATE", "NAVIGATOR", "WAYFINDER", "ACRUX", "GACRUX",
        "MIMOSA", "IMAI", "GINAN", "POLYNESIA", "PACIFIC", "CONSTELLATION"]

for order in perms:
    for near_first in (True, False):
        for inc_g in (True, False):
            for gf in (True, False):
                if not inc_g and gf:
                    continue
                s = make_strings(list(order), near_first, inc_g, gf)
                for k in range(26):
                    cand = caesar(s, k)
                    if is_word(cand) or splits_words(cand):
                        matches.append(("caesar", k, order, near_first, inc_g, gf, s, cand))
                    cand2 = caesar(atbash(s), k)
                    if is_word(cand2) or splits_words(cand2):
                        matches.append(("atbash+caesar", k, order, near_first, inc_g, gf, s, cand2))
                for key in keys:
                    cand = vigenere_dec(s, key)
                    if is_word(cand) or splits_words(cand):
                        matches.append(("vig", key, order, near_first, inc_g, gf, s, cand))

print(f"total matches: {len(matches)}")
for m in matches[:200]:
    print(f"{m[0]:12s} {m[1]!s:10s} order={''.join(m[2])} nf={m[3]} ig={m[4]} gf={m[5]} raw={m[6]} => {m[7]}")