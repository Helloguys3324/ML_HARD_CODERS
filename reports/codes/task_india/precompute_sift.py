#!/usr/bin/env python3
"""Precompute SIFT descriptors + gray thumbnails for all train images. Cached to disk."""
import os, glob, re, time, pickle
import cv2
import numpy as np

cv2.setNumThreads(4)
BASE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(BASE, 'data', 'train')

files = sorted(glob.glob(os.path.join(D, '*.jpg')))
names = [os.path.basename(f) for f in files]
print("n", len(names), flush=True)

sift = cv2.SIFT_create(nfeatures=3000)
out = {}
t0 = time.time()
for i, f in enumerate(files):
    n = os.path.basename(f)
    im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    h, w = im.shape[:2]
    s = 600 / max(h, w)
    if s < 1:
        im = cv2.resize(im, (max(1, int(w*s)), max(1, int(h*s))), interpolation=cv2.INTER_AREA)
    kp, des = sift.detectAndCompute(im, None)
    out[n] = dict(
        kp=[(k.pt[0], k.pt[1], k.size, k.angle, k.response) for k in kp] if kp else [],
        des=des.astype(np.float32) if des is not None else np.zeros((0, 128), np.float32),
    )
    if (i+1) % 500 == 0:
        print(i+1, f"{time.time()-t0:.1f}s", flush=True)

with open(os.path.join(BASE, 'sift_cache.pkl'), 'wb') as fp:
    pickle.dump(out, fp)
print("saved sift_cache.pkl", len(out), f"{time.time()-t0:.1f}s", flush=True)