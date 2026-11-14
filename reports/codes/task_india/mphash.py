#!/usr/bin/env python3
"""Multi-zoom pHash extraction + candidate analysis (development helper)."""
import os, sys, glob, numpy as np, cv2
from multiprocessing import Pool

cv2.setNumThreads(1)
FRACS = [1.0, 0.9, 0.84, 0.78, 0.72, 0.66, 0.60, 0.54]


def phash_from_img(img, ds=32, hs=8):
    img = cv2.resize(img, (ds, ds), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(img)
    low = d[:hs, :hs].copy()
    med = np.median(low[1:, :])
    bits = (low > med).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return np.uint64(val)


def feat_one(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hashes = []
    for f in FRACS:
        ch, cw = int(round(h * f)), int(round(w * f))
        y0 = (h - ch) // 2
        x0 = (w - cw) // 2
        hashes.append(phash_from_img(g[y0:y0 + ch, x0:x0 + cw]))
    return os.path.basename(path), np.array(hashes, dtype=np.uint64), h, w


def run(paths, out, workers=4):
    res = []
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(feat_one, paths, chunksize=8)):
            if r is not None:
                res.append(r)
    res.sort(key=lambda x: x[0])
    np.savez_compressed(out, names=np.array([r[0] for r in res]),
                        mph=np.stack([r[1] for r in res]),
                        dims=np.array([[r[2], r[3]] for r in res], dtype=np.int32))
    print('saved', out, len(res))


if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else '/home/kozak/mllm/MML/work/tasks/task_india/data/train'
    out = sys.argv[2] if len(sys.argv) > 2 else '/home/kozak/mllm/MML/work/solutions/task_india/mphash.npz'
    paths = sorted(glob.glob(os.path.join(root, '*.jpg')) + glob.glob(os.path.join(root, '*.png')))
    print(len(paths), 'images')
    run(paths, out)
