#!/usr/bin/env python3
"""Feature extraction for India duplicate-pair task (development helper)."""
import os, sys, glob, numpy as np, cv2, json
from multiprocessing import Pool

cv2.setNumThreads(1)


def phash(gray, hs=8, ds=32):
    img = cv2.resize(gray, (ds, ds), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(img)
    low = d[:hs, :hs].copy()
    med = np.median(low[1:, :])
    bits = (low > med).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def dhash(gray):
    img = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (img[:, 1:] > img[:, :-1]).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def ahash(gray):
    img = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    m = img.mean()
    bits = (img > m).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def feat_one(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ph = phash(gray)
    dh = dhash(gray)
    ah = ahash(gray)
    g32 = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    g32 = g32 - g32.mean()
    n = np.linalg.norm(g32)
    if n > 1e-6:
        g32 /= n
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return (os.path.basename(path), ph, dh, ah, g32.flatten(), hist.flatten().astype(np.float32), h, w)


def extract(paths, out, workers=4):
    res = []
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(feat_one, paths, chunksize=8)):
            if r is not None:
                res.append(r)
            if (i + 1) % 200 == 0:
                print(i + 1, '/', len(paths), flush=True)
    res.sort(key=lambda x: x[0])
    names = np.array([r[0] for r in res])
    ph = np.array([r[1] for r in res], dtype=np.uint64)
    dh = np.array([r[2] for r in res], dtype=np.uint64)
    ah = np.array([r[3] for r in res], dtype=np.uint64)
    g32 = np.stack([r[4] for r in res])
    hist = np.stack([r[5] for r in res])
    dims = np.array([[r[6], r[7]] for r in res], dtype=np.int32)
    np.savez_compressed(out, names=names, phash=ph, dhash=dh, ahash=ah,
                        g32=g32, hist=hist, dims=dims)
    print('saved', out, len(names))


if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else '/home/kozak/mllm/MML/work/tasks/task_india/data/train'
    out = sys.argv[2] if len(sys.argv) > 2 else '/home/kozak/mllm/MML/work/solutions/task_india/feats.npz'
    paths = sorted(glob.glob(os.path.join(root, '*.jpg')) + glob.glob(os.path.join(root, '*.png')))
    print(len(paths), 'images')
    extract(paths, out)
