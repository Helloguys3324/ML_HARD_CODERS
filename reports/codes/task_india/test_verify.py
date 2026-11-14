#!/usr/bin/env python3
"""Robust duplicate verifier v2: SIFT + RANSAC + CORRECT inverse-warp residual."""
import csv, re, cv2, os, glob
import numpy as np

D = 'data/train/'
pairs = []
with open('data/train-output.csv') as f:
    r = csv.reader(f); next(r)
    pairs = [(a, b) for a, b in r]
true = set(frozenset((a, b)) for a, b in pairs)

def load_gray(name, md=900):
    im = cv2.imread(D + name, cv2.IMREAD_GRAYSCALE)
    h, w = im.shape[:2]
    s = md / max(h, w)
    if s < 1:
        im = cv2.resize(im, (max(1,int(w*s)), max(1,int(h*s))), interpolation=cv2.INTER_AREA)
    return im

_sift = cv2.SIFT_create(nfeatures=4000)
IDX = dict(algorithm=1, trees=5)
SRCH = dict(checks=80)

def verify(a, b):
    ia = load_gray(a); ib = load_gray(b)
    kp1, d1 = _sift.detectAndCompute(ia, None)
    kp2, d2 = _sift.detectAndCompute(ib, None)
    if d1 is None or d2 is None:
        return dict(inl=0, good=0, ncc=-1, mae=999, ratio=0)
    flann = cv2.FlannBasedMatcher(IDX, SRCH)
    try:
        knn = flann.knnMatch(d1, d2, k=2)
    except Exception:
        return dict(inl=0, good=0, ncc=-1, mae=999, ratio=0)
    good = [m[0] for m in knn if len(m) >= 2 and m[0].distance < 0.75 * m[1].distance]
    if len(good) < 10:
        return dict(inl=0, good=len(good), ncc=-1, mae=999, ratio=0)
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None:
        return dict(inl=0, good=len(good), ncc=-1, mae=999, ratio=0)
    inl = int(mask.sum()) if mask is not None else 0
    # H maps  ia -> ib.  To align ib onto ia: warp ib by H^{-1}
    Hinv = np.linalg.inv(H)
    h, w = ia.shape
    warped = cv2.warpPerspective(ib, Hinv, (w, h))
    m = cv2.warpPerspective(np.full_like(ib, 255, np.uint8), Hinv, (w, h)) > 127
    if m.sum() < 2000:
        return dict(inl=inl, good=len(good), ncc=-1, mae=999, ratio=inl/len(good))
    x = ia[m].astype('float32'); y = warped[m].astype('float32')
    xn = (x - x.mean()) / (x.std()+1e-6); yn = (y - y.mean()) / (y.std()+1e-6)
    ncc = float((xn*yn).mean())
    mae = float(np.abs(x - y).mean())
    return dict(inl=inl, good=len(good), ncc=ncc, mae=mae, ratio=inl/len(good))

base = sorted(glob.glob(D + '*.jpg'))
names = [os.path.basename(f) for f in base]

targets = [
    ('img_000715.jpg','img_000716.jpg', True),
    ('img_001442.jpg','img_001443.jpg', True),
    ('img_000847.jpg','img_000848.jpg', True),
    ('img_003265.jpg','img_003266.jpg', True),
    ('img_002730.jpg','img_002731.jpg', True),
    ('img_001573.jpg','img_001574.jpg', False),
    ('img_003447.jpg','img_003448.jpg', False),
    ('img_003432.jpg','img_003433.jpg', False),
    ('img_000225.jpg','img_000226.jpg', False),
    ('img_000004.jpg','img_000005.jpg', True),
    ('img_000033.jpg','img_000034.jpg', True),
    ('img_003867.jpg','img_003868.jpg', False),
]
for a, b, lab in targets:
    r = verify(a, b)
    print(f"{a} {b} true={lab}: inl={r['inl']:4d} good={r['good']:4d} ratio={r['ratio']:.3f} ncc={r['ncc']:.4f} mae={r['mae']:.2f}")