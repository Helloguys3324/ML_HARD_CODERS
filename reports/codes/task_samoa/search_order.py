"""Explore all guide orders x Caesar (+ guide inclusion) with trigram scoring."""
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


def caesar(s, k):
    return "".join(chr((ord(c) - 65 - k) % 26 + 65) for c in s)


# companion groups keyed by guide letter (near->far)
COMPS = {
    'T': ['C', 'E', 'V'],
    'N': ['M', 'K', 'H'],
    'E': ['Y', 'Y'],
    'C': ['N', 'V'],
    'I': ['Z', 'U'],
}
GUIDES = ['T', 'N', 'E', 'C', 'I']


def build(order, near_first, inc_guides, guide_pos):
    parts = []
    for g in order:
        cs = COMPS[g] if near_first else COMPS[g][::-1]
        if inc_guides:
            seq = [g] + cs if guide_pos == 'first' else cs + [g]
        else:
            seq = cs
        parts.extend(seq)
    return "".join(parts)


results = []
for order in itertools.permutations(GUIDES):
    for near_first in (True, False):
        for inc_guides in (True, False):
            for gp in ('first', 'last'):
                if not inc_guides and gp == 'last':
                    continue
                s = build(order, near_first, inc_guides, gp)
                for k in range(26):
                    c = caesar(s, k)
                    results.append((score(c), list(order), near_first, inc_guides, gp, k, c))

results.sort(reverse=True)
seen = set()
shown = 0
for sc, order, nf, ig, gp, k, c in results:
    key = c
    if key in seen:
        continue
    seen.add(key)
    print(f"{sc:7.1f} order={''.join(order)} nf={nf} ig={ig} {gp}{k:3d} => {c}")
    shown += 1
    if shown >= 40:
        break