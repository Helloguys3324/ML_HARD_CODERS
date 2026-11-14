"""Comprehensive automated cipher search with trigram scoring."""
import glob, re, math, collections, sys

# --- build trigram model ---
tric = collections.Counter()
for f in glob.glob('/home/kozak/nltk_data/corpora/gutenberg/*.txt'):
    txt = open(f, encoding='utf-8', errors='ignore').read().lower()
    txt = re.sub(r'[^a-z ]', ' ', txt)
    for i in range(len(txt) - 2):
        t = txt[i:i + 3]
        if t.isalpha():
            tric[t] += 1
total = sum(tric.values())
logp = {k: math.log((v + 1) / (total + len(tric))) for k, v in tric.items()}
floor = math.log(1.0 / (total + len(tric)))


def score(s):
    s = s.lower()
    return sum(logp.get(s[i:i + 3], floor) for i in range(len(s) - 2))


WORDS = set()
for line in open('/home/kozak/nltk_data/corpora/words/en'):
    w = line.strip().lower()
    if w.isalpha() and 2 <= len(w) <= 12:
        WORDS.add(w)


def vig_dec(s, k):
    ks = [ord(c) - 65 for c in k.upper()]
    return "".join(chr((ord(c) - 65 - ks[i % len(ks)]) % 26 + 65) for i, c in enumerate(s))


def caesar(s, k):
    return "".join(chr((ord(c) - 65 - k) % 26 + 65) for c in s)


RAWS = {
    "gm_full": "TCEVNMKHEYYCNVIZU",
    "gm_comp": "CEVMKHYYNVZU",
    "ring_full": "TNECICMYNZEKYVUVH",
    "ring_comp": "CMYNZEKYVUVH",
    "grplast": "CEVTMKHNYYENVCZUI",
    "json": "NCEUVTCMYHKNZEVYI",
    "json_rev": "IYVEZNKHYMCTVUECN",
}

results = []
for rn, raw in RAWS.items():
    # Caesar
    for k in range(26):
        results.append((score(caesar(raw, k)), rn, f"caesar{k}", caesar(raw, k)))
    # Vigenere with dictionary word keys
    for key in WORDS:
        p = vig_dec(raw, key)
        results.append((score(p), rn, f"vig:{key}", p))

results.sort(reverse=True)
print("=== TOP 60 by trigram score ===")
seen = set()
shown = 0
for sc, rn, meth, plain in results:
    key = (rn, plain)
    if key in seen:
        continue
    seen.add(key)
    print(f"{sc:7.1f}  {rn:10s} {meth:18s} => {plain}")
    shown += 1
    if shown >= 60:
        break