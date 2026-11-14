#!/usr/bin/env python3
"""Evaluate SIFT verification on train candidate pairs."""
import numpy as np, csv, pickle, cv2, time, sys
from multiprocessing import Pool

cv2.setNumThreads(1)
base = '/home/kozak/mllm/MML/work/solutions/task_india/'

with open(base + 'sift.pkl', 'rb') as fh:
    SIFT = pickle.load(fh)
D = np.load(base + 'mphash.npz')
names = D['names'].astype(str)
dims = D['dims']
MIN = np.load(base + 'minzoom.npy')
idx = {n: i for i, n in enumerate(names)}
n = len(names)
ar = dims[:, 0] / dims[:, 1]
AR = np.minimum(ar[:, None], ar[None, :]) / np.maximum(ar[:, None], ar[None, :])
iu = np.triu_indices(n, 1)
mv = MIN[iu]; av = AR[iu]
cmask = (mv <= 20) & (av >= 0.98)
cand = np.stack(iu, 1)[cmask]
print('candidates', len(cand), flush=True)

rows = list(csv.DictReader(open(base + 'train-output.csv')))
truth = set()
for r in rows:
    if r['image_a'] in idx and r['image_b'] in idx:
        a, b = idx[r['image_a']], idx[r['image_b']]
        truth.add((min(a, b), max(a, b)))


def match_pair(ab):
    a, b = int(ab[0]), int(ab[1])
    na, nb = names[a], names[b]
    sa, sb = SIFT.get(na), SIFT.get(nb)
    if sa is None or sb is None:
        return a, b, 0, 0, 0.0, 0.0
    (sha, swa), pa, da = sa
    (shb, swb), pb, db = sb
    if len(da) < 8 or len(db) < 8:
        return a, b, 0, 0, 0.0, 0.0
    bf = cv2.BFMatcher(cv2.NORM_L2)
    m = bf.knnMatch(da, db, k=2)
    good = [x for x, y in m if x.distance < 0.75 * y.distance]
    if len(good) < 6:
        return a, b, len(good), 0, 0.0, 0.0
    src = np.float32([pa[g.queryIdx] for g in good]).reshape(-1, 1, 2)
    dst = np.float32([pb[g.trainIdx] for g in good]).reshape(-1, 1, 2)
    A, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                          ransacReprojThreshold=3.0, maxIters=2000)
    if A is None or mask is None:
        return a, b, len(good), 0, 0.0, 0.0
    inl = int(mask.sum())
    s = float(np.sqrt(A[0, 0] ** 2 + A[1, 0] ** 2))
    return a, b, len(good), inl, inl / max(len(good), 1), s


if __name__ == '__main__':
    t0 = time.time()
    res = []
    with Pool(4) as pool:
        for i, r in enumerate(pool.imap_unordered(match_pair, cand, chunksize=8)):
            res.append(r)
            if (i + 1) % 500 == 0:
                print(i + 1, '/', len(cand), time.time() - t0, flush=True)
    with open(base + 'cand_sift.pkl', 'wb') as fh:
        pickle.dump(res, fh)
    print('done', time.time() - t0)
