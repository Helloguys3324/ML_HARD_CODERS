"""Final broad search: many orderings x {Caesar, Atbash+Caesar, Affine} trigram-scored."""
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


def affine_dec(s, a, b):
    ainv = None
    for x in range(26):
        if (a * x) % 26 == 1:
            ainv = x
    return "".join(chr((ainv * (ord(c) - 65 - b)) % 26 + 65) for c in s)


# per-guide companions (near->far) and angular (clockwise) order
COMPS = {'T': ['C', 'E', 'V'], 'N': ['M', 'K', 'H'], 'E': ['Y', 'Y'], 'C': ['N', 'V'], 'I': ['Z', 'U']}
# angular order (atan2 ascending) computed earlier
ANG = {'T': ['E', 'V', 'C'], 'N': ['M', 'K', 'H'], 'E': ['Y', 'Y'], 'C': ['N', 'V'], 'I': ['Z', 'U']}
GUIDES = ['T', 'N', 'E', 'C', 'I']


def build(order, comps, inc_guides, guide_pos, ring_order):
    parts = []
    for g in order:
        cs = comps[g]
        if inc_guides:
            seq = [g] + cs if guide_pos == 'first' else cs + [g]
        else:
            seq = cs
        parts.extend(seq)
    s = "".join(parts)
    if ring_order == 'ring':  # ring-major: all nearest, then all middle, then all far
        byring = {}
        for g in order:
            for r, c in enumerate(comps[g]):
                byring.setdefault(r, []).append(c)
        s = (("".join(order)) if inc_guides else "") + "".join("".join(byring[r]) for r in sorted(byring))
    return s


results = []
orders = list(itertools.permutations(GUIDES))
for order in orders:
    for comps_name, comps in (('dist', COMPS), ('ang', ANG)):
        for inc_guides in (True, False):
            for gp in ('first', 'last'):
                if not inc_guides and gp == 'last':
                    continue
                for ring in ('guidemajor', 'ring'):
                    s = build(order, comps, inc_guides, gp, ring)
                    for k in range(26):
                        results.append((score(caesar(s, k)), 'caesar', ''.join(order), comps_name, inc_guides, gp, ring, k, caesar(s, k)))
                        s2 = caesar(s, k)  # placeholder
                    # atbash + caesar
                    at = caesar(s[::-1], 0)[::-1] if False else "".join(chr(25 - (ord(c) - 65) + 65) for c in s)
                    for k in range(26):
                        results.append((score(caesar(at, k)), 'atbash+caesar', ''.join(order), comps_name, inc_guides, gp, ring, k, caesar(at, k)))
                    # affine
                    for a in (1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25):
                        for b in range(26):
                            results.append((score(affine_dec(s, a, b)), 'affine', ''.join(order), comps_name, inc_guides, gp, ring, (a, b), affine_dec(s, a, b)))

results.sort(reverse=True)
seen = set()
shown = 0
for sc, meth, o, cname, ig, gp, ring, key, plain in results:
    if plain in seen:
        continue
    seen.add(plain)
    print(f"{sc:7.1f} {meth:12s} ord={o} {cname} ig={ig}{gp:5s}{ring:11s} k={key} => {plain}")
    shown += 1
    if shown >= 50:
        break