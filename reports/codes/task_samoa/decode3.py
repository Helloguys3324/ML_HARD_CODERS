"""Vigenere crack: try every reasonable dictionary word as key."""
import re

WORDS = set()
for line in open("/home/kozak/nltk_data/corpora/words/en"):
    w = line.strip().lower()
    if w.isalpha() and 2 <= len(w) <= 10:
        WORDS.add(w)

RAW = {
    "gm_full": "TCEVNMKHEYYCNVIZU",
    "gm_comp": "CEVMKHYYNVZU",
    "ring_full": "TNECICMYNZEKYVUVH",
    "ring_comp": "CMYNZEKYVUVH",
    "grplast": "CEVTMKHNYYENVCZUI",
    "guides": "TNECI",
}


def vig_dec(s, key):
    return "".join(chr((ord(c) - 65 - (ord(key[i % len(key)]) - 65)) % 26 + 65) for i, c in enumerate(s))


def phrase_ok(s):
    ls = s.lower()
    if ls in WORDS:
        return True
    n = len(s)
    # split into <=3 words each len>=3
    def rec(i, cnt):
        if i == n:
            return cnt >= 1
        if cnt >= 3:
            return False
        for j in range(i + 3, min(i + 12, n) + 1):
            if s[i:j] in WORDS and rec(j, cnt + 1):
                return True
        return False
    return rec(0, 0)


keys_sorted = sorted(WORDS, key=len)
# prioritize thematic/common keys, then all
thematic = ["crux", "cross", "south", "southern", "home", "star", "stars", "five", "greek",
            "watch", "foundation", "basis", "base", "sailor", "ocean", "wave", "wind",
            "voyage", "sail", "sails", "canoe", "way", "navigation", "navigate", "compass",
            "anchor", "island", "samoa", "apia", "pacific", "polynesia", "light", "lights",
            "sky", "night", "homeward", "guide", "guides", "north", "south", "east", "west",
            "acrux", "gacrux", "mimosa", "imai", "ginan", "centi", "tncii", "alpha", "beta",
            "gamma", "delta", "epsilon", "letter", "letters", "word", "message", "secret"]

found = []
for raw_name, cipher in RAW.items():
    for key in thematic + keys_sorted:
        plain = vig_dec(cipher, key)
        if phrase_ok(plain):
            found.append((raw_name, key, plain))
    print(f"{raw_name}: done, running found list has {len(found)}")

print("\n=== RESULTS ===")
for raw_name, key, plain in found:
    print(f"{raw_name:10s} key={key!r:14s} => {plain}")