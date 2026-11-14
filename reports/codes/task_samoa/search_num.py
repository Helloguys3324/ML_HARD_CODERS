"""Numeric-key polyalphabetic search (Gronsfeld/Beaufort/progressive) with trigram scoring."""
import math, itertools, glob, re, collections

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


RAW = "TCEVNMKHEYYCNVIZU"


def gronsfeld(s, keys, op=lambda c, k: c - k):
    return "".join(chr(op(ord(c) - 65, keys[i % len(keys)]) % 26 + 65) for i, c in enumerate(s))


results = []

# 1. Gronsfeld with numeric keys length 1..4, values 0..25
for L in range(1, 5):
    for keys in itertools.product(range(26), repeat=L):
        p = gronsfeld(RAW, keys, lambda c, k: c - k)  # standard decrypt
        results.append((score(p), 'gron', keys, p))
        p2 = gronsfeld(RAW, keys, lambda c, k: k - c)  # beaufort decrypt
        results.append((score(p2), 'beau', keys, p2))

# 2. progressive Caesar shift = b + i*s (i=index)
for b in range(26):
    for s in range(26):
        p = "".join(chr((ord(c) - 65 - (b + i * s)) % 26 + 65) for i, c in enumerate(RAW))
        results.append((score(p), 'prog', (b, s), p))

results.sort(reverse=True)
seen = set()
shown = 0
for sc, meth, keys, p in results:
    if p in seen:
        continue
    seen.add(p)
    print(f"{sc:7.1f} {meth:5s} key={keys} => {p}")
    shown += 1
    if shown >= 40:
        break