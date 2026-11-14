#!/usr/bin/env python3
"""Match ambiguous pairs via SIFT and compute discriminative features. Caches results."""
import os, glob, re, time, pickle, csv
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

print("loading sift cache...", flush=True)
with open(os.path.join(BASE, 'sift_cache.pkl'), 'rb') as fp:
    scache = pickle.load(fp)

# gray thumbnails (96) + shapes + color hists
S = 96
gray = {}
shape = {}
for i, n in enumerate(names):
    im = cv2.imread(os.path.join(D, n), cv2.IMREAD_GRAYSCALE)
    shape[n] = (im.shape[1], im.shape[0])
    g = cv2.resize(im, (S, S), interpolation=cv2.INTER_AREA).astype('float32')
    gray[n] = (g - g.mean()) / (g.std() + 1e-6)

def ncc_direct(a, b):
    return float((gray[a] * gray[b]).mean())

def ardiff(a, b):
    wa, ha = shape[a]; wb, hb = shape[b]
    return abs(wa/ha - wb/hb) / max(wa/ha, wb/hb)

FLANN = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=80))

def kp_from_idx(cache, name):
    rec = cache[name]
    return rec['kp'], rec['des']

def warp_features(a, b):
    kp1, d1 = kp_from_idx(scache, a)
    kp2, d2 = kp_from_idx(scache, b)
    res = dict(nmatch=0, ninl=0, ratio=0.0, ncc=-1, ncc_cc=-1, mae=-1, mae_cc=-1,
               rot=0.0, scale=0.0, shear=0.0, H=None)
    if len(kp1) < 8 or len(kp2) < 8:
        return res
    try:
        knn = FLANN.knnMatch(d1, d2, k=2)
    except Exception:
        return res
    good = [m[0] for m in knn if len(m) >= 2 and m[0].distance < 0.75 * m[1].distance]
    res['nmatch'] = len(good)
    if len(good) < 10:
        return res
    src = np.float32([kp1[m.queryIdx][:2] for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx][:2] for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None:
        return res
    ninl = int(mask.sum()) if mask is not None else 0
    res['ninl'] = ninl
    res['ratio'] = ninl / max(1, len(good))
    # decompose H -> affine part
    A = H[:2, :2]
    U, s, Vt = np.linalg.svd(A)
    res['scale'] = float(s[0])
    res['shear'] = float(abs(s[0] - s[1]) / max(1e-6, s[0]))
    res['rot'] = float(np.degrees(np.arctan2(U[1,0], U[0,0])))
    # load full gray images (already at 600 in cache? no, cache has kp only). reload original at md
    ia = cv2.imread(os.path.join(D, a), cv2.IMREAD_GRAYSCALE)
    ib = cv2.imread(os.path.join(D, b), cv2.IMREAD_GRAYSCALE)
    md = 600
    for name, im in [(a, ia), (b, ib)]:
        h, w = im.shape[:2]
        sc = md / max(h, w)
        if sc < 1:
            if name == a: ia = cv2.resize(im, (max(1,int(w*sc)), max(1,int(h*sc))), interpolation=cv2.INTER_AREA)
            else: ib = cv2.resize(im, (max(1,int(w*sc)), max(1,int(h*sc))), interpolation=cv2.INTER_AREA)
        else:
            if name == a: ia = im
            else: ib = im
    # H maps a->b (pts). To align b onto a: warp b by H^-1
    Hinv = np.linalg.inv(H)
    h, w = ia.shape
    warped = cv2.warpPerspective(ib, Hinv, (w, h))
    msk = cv2.warpPerspective(np.full_like(ib, 255, np.uint8), Hinv, (w, h)) > 127
    if msk.sum() < 2000:
        return res
    x = ia[msk].astype('float32'); y = warped[msk].astype('float32')
    xn = (x - x.mean()) / (x.std() + 1e-6); yn = (y - y.mean()) / (y.std() + 1e-6)
    res['ncc'] = float((xn * yn).mean())
    res['mae'] = float(np.abs(x - y).mean())
    # linear color correction y = a*x + b
    A2 = np.vstack([x, np.ones_like(x)]).T
    coef, _, _, _ = np.linalg.lstsq(A2, y, rcond=None)
    y_hat = coef[0] * x + coef[1]
    res['gain'] = float(coef[0]); res['offset'] = float(coef[1])
    res['mae_cc'] = float(np.abs(y - y_hat).mean())
    yhn = (y_hat - y_hat.mean()) / (y_hat.std() + 1e-6)
    res['ncc_cc'] = float((xn * yhn).mean())
    return res

def is_ambig(a, b, lo=0.05, hi=0.85):
    return ardiff(a, b) < 0.02 and lo < ncc_direct(a, b) < hi

# ambiguous pairs (gap1 + gap2)
amb = []
for i in range(len(names) - 1):
    a, b = names[i], names[i+1]
    if is_ambig(a, b):
        amb.append((a, b))
for i in range(len(names) - 2):
    a, b = names[i], names[i+2]
    if is_ambig(a, b):
        amb.append((a, b))

print("ambiguous pairs:", len(amb), flush=True)
rows = {}
t0 = time.time()
for k, (a, b) in enumerate(amb):
    wf = warp_features(a, b)
    wf['label'] = 1 if frozenset((a, b)) in true else 0
    wf['ncc_direct'] = ncc_direct(a, b)
    wf['ardiff'] = ardiff(a, b)
    rows[(a, b)] = wf
    if (k+1) % 100 == 0:
        print(k+1, f"{time.time()-t0:.1f}s", flush=True)

with open(os.path.join(BASE, 'warp_features.pkl'), 'wb') as fp:
    pickle.dump(rows, fp)
print("saved warp_features.pkl", len(rows), flush=True)