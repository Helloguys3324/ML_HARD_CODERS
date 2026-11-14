#!/usr/bin/env python3
"""Compute direct NCC (best over 4 rotations x 2 flips) for ALL candidate pairs. Save to pkl."""
import csv, os, glob, re, pickle
import numpy as np
import cv2

cv2.setNumThreads(4)
BASE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(BASE, 'data', 'train')

pairs = []
with open(os.path.join(BASE, 'data', 'train-output.csv')) as f:
    r = csv.reader(f); next(r)
    pairs = [(a, b) for a, b in r]
true = set(frozenset((a, b)) for a, b in pairs)

files = sorted(glob.glob(os.path.join(D, '*.jpg')))
names = [os.path.basename(f) for f in files]

S = 64
imgs = {}
for n in names:
    im = cv2.imread(os.path.join(D, n), cv2.IMREAD_GRAYSCALE)
    h, w = im.shape[:2]
    l = max(h, w)
    sq = np.full((l, l), int(im.mean()), dtype=im.dtype)
    y0 = (l - h) // 2; x0 = (l - w) // 2
    sq[y0:y0+h, x0:x0+w] = im
    sq = cv2.resize(sq, (S, S), interpolation=cv2.INTER_AREA).astype('float32')
    sig = sq.std()
    imgs[n] = (sq - sq.mean()) / (sig + 1e-6)

cands = [(names[i], names[i+1]) for i in range(len(names)-1)] + \
        [(names[i], names[i+2]) for i in range(len(names)-2)]

out = {}
for a, b in cands:
    x = imgs[a]; y = imgs[b]
    best_ncc = -2.0; best_mae = 999.0
    for rot in [0, 1, 2, 3]:
        yr = np.rot90(y, rot)
        for flip in [False, True]:
            yrf = yr[:, ::-1] if flip else yr
            c = float((x * yrf).mean())
            if c > best_ncc:
                best_ncc = c
                best_mae = float(np.abs(x - yrf).mean())
    out[(a, b)] = dict(ncc=best_ncc, mae=best_mae,
                       label=1 if frozenset((a, b)) in true else 0)

pickle.dump(out, open(os.path.join(BASE, 'ncc_features.pkl'), 'wb'))
print("saved", len(out))

v = np.array([[o['ncc'] for o in out.values()], [o['label'] for o in out.values()]]).T
for lo, hi in [(0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.0), (0.3, 1.0)]:
    m = (v[:, 0] >= lo) & (v[:, 0] < hi)
    print(f"NCC [{lo:.1f},{hi:.1f}): n={m.sum()} pos={(v[m,1]==1).sum()}")