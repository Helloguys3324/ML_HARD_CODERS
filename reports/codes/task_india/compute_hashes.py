#!/usr/bin/env python3
"""Compute perceptual hashes for all images and dump to a pickle for fast iteration."""
import pickle, glob, os, time
import numpy as np
import cv2

from duputils import load_gray, load_rgb, dhash, ahash, phash, color_hist, rgb_mean

files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), 'data', 'train', '*.jpg')))
print("n files", len(files))

records = {}
t0 = time.time()
for i, f in enumerate(files):
    name = os.path.basename(f)
    gray = load_gray(f, max_dim=256)
    rgb = load_rgb(f, max_dim=256)
    if gray is None:
        continue
    records[name] = dict(
        dh=dhash(gray), ah=ahash(gray), ph=phash(gray),
        hist=color_hist(rgb, bins=6).tolist(),
        mean=rgb_mean(rgb).tolist(),
        shape=[gray.shape[0], gray.shape[1]],
        size=os.path.getsize(f),
    )
    if i % 500 == 0:
        print(i, f"{time.time()-t0:.1f}s", flush=True)

with open(os.path.join(os.path.dirname(__file__), 'hashes.pkl'), 'wb') as fp:
    pickle.dump(records, fp)
print("saved", len(records), f"{time.time()-t0:.1f}s")