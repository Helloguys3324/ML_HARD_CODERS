#!/usr/bin/env python3
"""SIFT+homography+aligned-NCC verification for candidate pairs, using cached descriptors."""
import os, glob, re, time, pickle, csv, sys
import numpy as np
import cv2

cv2.setNumThreads(4)
BASE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(BASE, 'data', 'train')
MD = 600  # match precompute_sift.py max_dim

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

cands = [(names[i], names[i+1]) for i in range(len(names)-1)] + \
        [(names[i], names[i+2]) for i in range(len(names)-2)]

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else len(cands)
cands = cands[:LIMIT]

FLANN = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=80))

def load_gray(n):
    im = cv2.imread(os.path.join(D, n), cv2.IMREAD_GRAYSCALE)
    h, w = im.shape[:2]
    s = MD / max(h, w)
    if s < 1:
        im = cv2.resize(im, (max(1, int(w*s)), max(1, int(h*s))), interpolation=cv2.INTER_AREA)
    return im

def knp(name):
    rec = scache[name]
    kp = [cv2.KeyPoint(x=k[0], y=k[1], size=k[2], angle=k[3], response=k[4]) for k in rec['kp']]
    return kp, rec['des']

def verify(a, b):
    res = dict(nmatch=0, ninl=0, ratio=0.0, ncc=-1.0, ncc_cc=-1.0, mae=999.0, mae_cc=999.0,
               scale=0.0, shear=0.0, rot=0.0)
    kp1, d1 = knp(a); kp2, d2 = knp(b)
    if len(kp1) < 8 or len(kp2) < 8 or d1.shape[0] < 4 or d2.shape[0] < 4:
        return res
    try:
        knn = FLANN.knnMatch(d1, d2, k=2)
    except Exception:
        return res
    good = [m[0] for m in knn if len(m) >= 2 and m[0].distance < 0.75 * m[1].distance]
    res['nmatch'] = len(good)
    if len(good) < 10:
        return res
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None:
        return res
    ninl = int(mask.sum()) if mask is not None else 0
    res['ninl'] = ninl
    res['ratio'] = ninl / max(1, len(good))
    A = H[:2, :2]
    U, s, Vt = np.linalg.svd(A)
    res['scale'] = float(s[0]); res['shear'] = float(abs(s[0]-s[1])/max(1e-6, s[0]))
    res['rot'] = float(np.degrees(np.arctan2(U[1,0], U[0,0])))
    ia = load_gray(a); ib = load_gray(b)
    Hinv = np.linalg.inv(H)
    h, w = ia.shape
    warped = cv2.warpPerspective(ib, Hinv, (w, h))
    m = cv2.warpPerspective(np.full_like(ib, 255, np.uint8), Hinv, (w, h)) > 127
    if m.sum() < 2000:
        return res
    x = ia[m].astype('float32'); y = warped[m].astype('float32')
    xn = (x - x.mean()) / (x.std() + 1e-6); yn = (y - y.mean()) / (y.std() + 1e-6)
    res['ncc'] = float((xn * yn).mean())
    res['mae'] = float(np.abs(x - y).mean())
    A2 = np.vstack([x, np.ones_like(x)]).T
    coef, _, _, _ = np.linalg.lstsq(A2, y, rcond=None)
    y_hat = coef[0] * x + coef[1]
    res['gain'] = float(coef[0]); res['offset'] = float(coef[1])
    res['mae_cc'] = float(np.abs(y - y_hat).mean())
    yhn = (y_hat - y_hat.mean()) / (y_hat.std() + 1e-6)
    res['ncc_cc'] = float((xn * yhn).mean())
    return res

rows = {}
t0 = time.time()
for k, (a, b) in enumerate(cands):
    v = verify(a, b)
    v['label'] = 1 if frozenset((a, b)) in true else 0
    rows[(a, b)] = v
    if (k+1) % 200 == 0:
        print(f"{k+1}/{len(cands)} {time.time()-t0:.0f}s", flush=True)

outf = os.path.join(BASE, 'warp_features.pkl')
with open(outf, 'wb') as fp:
    pickle.dump(rows, fp)
print(f"saved {outf} n={len(rows)}  {time.time()-t0:.0f}s", flush=True)