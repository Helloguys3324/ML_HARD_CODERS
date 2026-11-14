#!/usr/bin/env python3
"""Fast analysis of true vs false adjacent pairs using numpy NCC + MAE at 64x64."""
import csv, re, cv2, os, glob
import numpy as np

D = 'data/train/'
pairs = []
with open('data/train-output.csv') as f:
    r = csv.reader(f); next(r)
    pairs = [(a, b) for a, b in r]
true = set(frozenset((a, b)) for a, b in pairs)

files = sorted(glob.glob(D + '*.jpg'))
names = [os.path.basename(f) for f in files]
S = 64
print("loading...")
imgs = {}
for n in names:
    im = cv2.imread(D + n, cv2.IMREAD_GRAYSCALE)
    imgs[n] = cv2.resize(im, (S, S), interpolation=cv2.INTER_AREA).astype('float32')
# normalize each to zero-mean unit-std
for n in names:
    x = imgs[n]; imgs[n] = (x - x.mean()) / (x.std() + 1e-6)

def feats(a, b):
    x = imgs[a]; y = imgs[b]
    ncc = float((x * y).mean())
    mae = float(np.abs(x - y).mean())
    return ncc, mae

import collections
cands_1 = [(names[i], names[i+1]) for i in range(len(names)-1)]
cands_2 = [(names[i], names[i+2]) for i in range(len(names)-2)]

rows = []
stats = collections.defaultdict(list)
for a, b in cands_1 + cands_2:
    lab = 'T' if frozenset((a, b)) in true else 'F'
    ncc, mae = feats(a, b)
    stats[lab].append((ncc, mae))

for lab in ['T', 'F']:
    arr = np.array(stats[lab])
    print(f"label {lab}: n={len(arr)}")
    for ci, col in enumerate(['NCC', 'MAE(std)']):
        v = arr[:, ci]
        print(f"  {col}: min {v.min():.4f} p1 {np.percentile(v,1):.4f} p5 {np.percentile(v,5):.4f} med {np.median(v):.4f} p95 {np.percentile(v,95):.4f} p99 {np.percentile(v,99):.4f} max {v.max():.4f}")

for gap, cs in [('gap1', cands_1), ('gap2', cands_2)]:
    ts = [feats(a,b) for a,b in cs if frozenset((a,b)) in true]
    fs = [feats(a,b) for a,b in cs if frozenset((a,b)) not in true]
    print(f"\n{gap}: true={len(ts)} false={len(fs)}")
    ts = np.array(ts) if ts else np.zeros((0,2))
    fs = np.array(fs) if fs else np.zeros((0,2))
    print(f"  TRUE  NCC med {np.median(ts[:,0]):.4f} min {ts[:,0].min():.4f}")
    print(f"  FALSE NCC med {np.median(fs[:,0]):.4f} max {fs[:,0].max():.4f}")