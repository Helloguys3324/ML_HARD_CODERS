#!/usr/bin/env python3
"""Comprehensive feature separation analysis on train gap1+gap2 candidate pairs."""
import csv, os, glob, re
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
print("n images", len(names))

S = 64
# load canonical grayscale thumbs at SxS, plus up to 4 rotations handled by trying
imgs = {}
shapes = {}
for n in names:
    im = cv2.imread(os.path.join(D, n), cv2.IMREAD_GRAYSCALE)
    h, w = im.shape[:2]
    shapes[n] = (w, h)
    # pad to square before resize to preserve aspect (avoid stretch)
    l = max(h, w)
    sq = np.full((l, l), im.mean(), dtype=im.dtype)
    y0 = (l - h)//2; x0 = (l - w)//2
    sq[y0:y0+h, x0:x0+w] = im
    sq = cv2.resize(sq, (S, S), interpolation=cv2.INTER_AREA).astype('float32')
    sig = sq.std()
    imgs[n] = (sq - sq.mean()) / (sig + 1e-6)

def ncc(x, y):
    return float((x * y).mean())

def mae_std(x, y):
    return float(np.abs(x - y).mean())

# candidate pairs: gap1 + gap2
cands = [(names[i], names[i+1]) for i in range(len(names)-1)] + \
        [(names[i], names[i+2]) for i in range(len(names)-2)]

def feats_direct(a, b):
    x = imgs[a]; y = imgs[b]
    best_ncc = -2; best_mae = 999; best_rot = -1
    for rot in [0, 1, 2, 3]:
        yr = np.rot90(y, rot)
        for flip in [False, True]:
            yrf = yr[:, ::-1] if flip else yr
            c = ncc(x, yrf)
            if c > best_ncc:
                best_ncc = c; best_mae = mae_std(x, yrf); best_rot = rot
    return best_ncc, best_mae

rows = []
for a, b in cands:
    lab = 1 if frozenset((a, b)) in true else 0
    bn, bm = feats_direct(a, b)
    rows.append((lab, bn, bm))

rows = np.array(rows)
T = rows[rows[:,0]==1]; F = rows[rows[:,0]==0]
print(f"total {len(rows)}  True {len(T)}  False {len(F)}")
for lab, A in [('TRUE', T), ('FALSE', F)]:
    for ci, nm in enumerate(['NCC64(best4rot)','MAE64']):
        v = A[:, ci+1]
        print(f"  {lab} {nm}: min {v.min():.4f} p1 {np.percentile(v,1):.4f} p5 {np.percentile(v,5):.4f} p50 {np.median(v):.4f} p95 {np.percentile(v,95):.4f} p99 {np.percentile(v,99):.4f} max {v.max():.4f}")

# find NCC threshold for F1
v = rows[:, 1]; lab = rows[:, 0]
best = None
for th in np.arange(-0.5, 1.501, 0.01):
    pred = v > th
    tp = ((pred==1)&(lab==1)).sum(); fp = ((pred==1)&(lab==0)).sum()
    fn = ((pred==0)&(lab==1)).sum()
    prec = tp/max(1,tp+fp); rec = tp/max(1,tp+fn)
    f1 = 2*prec*rec/max(1e-9,prec+rec)
    if best is None or f1 > best[0]:
        best = (f1, th, prec, rec)
print(f"\nBEST direct NCC: F1={best[0]:.4f} thresh={best[1]:.2f} prec={best[2]:.4f} rec={best[3]:.4f}")