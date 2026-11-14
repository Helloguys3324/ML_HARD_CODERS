"""Big structured search for the flag word."""
import math
import itertools

STARS = [
("N",40,88,10),("C",97.454,197.819,3.5),("E",96,28,10),("U",110.092,137.003,3.5),("V",125.985,169.57,3.5),
("T",100,180,10),("C",160,76,10),("M",22.828,82.603,3.5),("Y",112.355,20.482,3.5),("H",46.086,115.331,3.5),
("K",56.943,72.446,3.5),("N",176.997,70.076,3.5),("Z",145.319,142.108,3.5),("E",81.908,165.799,3.5),
("V",155.696,98.594,3.5),("Y",93.871,50.901,3.5),("I",132,130,7)]
GUIDES = [s for s in STARS if s[3] > 6]
COMPS = [s for s in STARS if s[3] <= 6]

WORDS = set()
for line in open("/home/kozak/nltk_data/corpora/words/en"):
    w = line.strip().lower()
    if w.isalpha() and len(w) >= 3:
        WORDS.add(w)

def hyp(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])

def nearest_guide(c):
    return min(GUIDES, key=lambda g: hyp((g[1], g[2]), (c[1], c[2])))

def caesar(s, k): return "".join(chr((ord(c)-65-k)%26+65) for c in s)
def vig_dec(s, key):
    o=[]
    for i,c in enumerate(s):
        o.append(chr((ord(c)-65-(ord(key[i%len(key)])-65))%26+65))
    return "".join(o)
def atbash(s): return "".join(chr(25-(ord(c)-65)+65) for c in s)

def phrase_ok(s):
    ls = s.lower()
    if ls in WORDS: return True
    n=len(s)
    # at most 3 words, each len>=3
    def rec(i, cnt):
        if i==n: return cnt>=1
        if cnt>=3: return False
        for j in range(i+3, min(i+12, n)+1):
            if s[i:j].lower() in WORDS and rec(j, cnt+1):
                return True
        return False
    return rec(0,0)

# Build different orderings
def build():
    # guide-major near->far
    order_greek = ['T','N','E','C','I']
    gmap = {g[0]: g for g in GUIDES}  # first char as key (unique)
    gm = {}
    for c in COMPS:
        g = nearest_guide(c)[0]
        gm.setdefault(g, []).append(c)
    for g in gm:
        gm[g].sort(key=lambda c: hyp((gmap[g][1],gmap[g][2]),(c[1],c[2])))
    rings = {}
    for c in COMPS:
        g = nearest_guide(c)[0]
        r = round(hyp((gmap[g][1],gmap[g][2]),(c[1],c[2])))
        rings.setdefault(r, []).append((g, c[0]))
    # guide-major string
    gm_s = "".join(g + "".join(c[0] for c in gm.get(g, [])) for g in order_greek)
    gm_comp = "".join("".join(c[0] for c in gm.get(g, [])) for g in order_greek)
    # ring-major (18,23,28) in guide order
    ring_s = "".join("".join(c[0] for g, c in sorted(v, key=lambda x: order_greek.index(x[0]))) for k, v in sorted(rings.items()))
    ring_s_full = "".join(order_greek) + ring_s
    return gm_s, gm_comp, ring_s, ring_s_full

gm_s, gm_comp, ring_s, ring_s_full = build()
print("guidemajor full:", gm_s)
print("guidemajor comp:", gm_comp)
print("ringmajor comp:", ring_s)
print("ringmajor full:", ring_s_full)

KEYS = ["CRUX","ACRUX","GACRUX","MIMOSA","IMAI","GINAN","CENTI","TNECI","TCENI","FIVE","SAMOA","APIA",
        "SIGMOID","FOUNDATION","GREEK","WATCH","CROSS","SOUTH","SOUTHERN","HOME","HOMEWARD","NAVIGATE",
        "NAVIGATOR","WAYFINDER","VOYAGE","SAILOR","MARINER","STAR","STARS","OCEAN","PACIFIC","POLYNESIA",
        "CANOE","COMPASS","ANCHOR","ALPHA","BETA","GAMMA","DELTA","EPSILON","GREEKWATCH","RWB"]

allraw = {
 "gm_full": gm_s,
 "gm_comp": gm_comp,
 "ring_comp": ring_s,
 "ring_full": ring_s_full,
}

found = []
for name, raw in allraw.items():
    for k in range(26):
        c = caesar(raw, k)
        if phrase_ok(c):
            found.append((name, f"caesar{k}", c))
    c = atbash(raw)
    if phrase_ok(c): found.append((name, "atbash", c))
    for key in KEYS:
        c = vig_dec(raw, key)
        if phrase_ok(c):
            found.append((name, f"vig-{key}", c))
        c2 = vig_dec(atbash(raw), key)
        if phrase_ok(c2):
            found.append((name, f"atbash+vig-{key}", c2))

print(f"\nTOTAL FOUND: {len(found)}")
for name, method, res in found:
    print(f"{name:10s} {method:20s} => {res}")