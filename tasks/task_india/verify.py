#!/usr/bin/env python3
"""ORB verify all candidates with T<=13 and save; then inspect hard FPs."""
import numpy as np, csv, cv2, time, pickle, sys
from multiprocessing import Pool

cv2.setNumThreads(1)
base = '/home/kozak/mllm/MML/work/solutions/task_india/'
ROOT = '/home/kozak/mllm/MML/work/tasks/task_india/data/train/'
D = np.load(base + 'mphash.npz'); names = D['names'].astype(str); idx = {x: i for i, x in enumerate(names)}
S = np.load(base + 'dirzoom.npy'); dims = D['dims']; n = len(names)
ar = dims[:, 0] / dims[:, 1]
AR = np.minimum(ar[:, None], ar[None, :]) / np.maximum(ar[:, None], ar[None, :])
iu = np.triu_indices(n, 1); mv = S[iu]; av = AR[iu]
TMAX = float(sys.argv[1]) if len(sys.argv) > 1 else 13
cand = np.stack(iu, 1)[(mv <= TMAX) & (av >= 0.97)]
rows = list(csv.DictReader(open(base + 'train-output.csv')))
truth = set((min(idx[r['image_a']], idx[r['image_b']]), max(idx[r['image_a']], idx[r['image_b']]))
            for r in rows if r['image_a'] in idx and r['image_b'] in idx)
print('candidates', len(cand), flush=True)


def work(ab):
    a, b = int(ab[0]), int(ab[1])
    orb = cv2.ORB_create(nfeatures=400, fastThreshold=10)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def feat(nm):
        im = cv2.imread(ROOT + nm, 0)
        h, w = im.shape
        if max(h, w) > 600:
            s = 600 / max(h, w)
            im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        kp, des = orb.detectAndCompute(im, None)
        pts = np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32)
        return im, pts, (des if des is not None else np.zeros((0, 32), np.uint8))

    ia, pa, da = feat(names[a]); ib, pb, db = feat(names[b])
    m = bf.knnMatch(da, db, k=2)
    good = [x for x, y in m if x.distance < 0.75 * y.distance]
    if len(good) < 6:
        return (a, b, len(good), 0, 0.0, 0.0, 0.0)
    src = np.float32([pa[g.queryIdx] for g in good]).reshape(-1, 1, 2)
    dst = np.float32([pb[g.trainIdx] for g in good]).reshape(-1, 1, 2)
    A, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                          ransacReprojThreshold=3.0, maxIters=2000)
    if A is None or mask is None:
        return (a, b, len(good), 0, 0.0, 0.0, 0.0)
    inl = int(mask.sum()); s = float(np.sqrt(A[0, 0] ** 2 + A[1, 0] ** 2))
    # NCC after warping a -> b
    hb, wb = ib.shape
    warp = cv2.warpAffine(ia, A, (wb, hb), flags=cv2.INTER_LINEAR, borderValue=0)
    maska = cv2.warpAffine(np.ones_like(ia), A, (wb, hb), flags=cv2.INTER_NEAREST, borderValue=0) > 0
    if maska.sum() < 100:
        ncc = 0.0
    else:
        va = warp[maska].astype(np.float32); vb = ib[maska].astype(np.float32)
        va -= va.mean(); vb -= vb.mean()
        ncc = float((va * vb).sum() / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-6))
    return (a, b, len(good), inl, inl / max(len(good), 1), s, ncc)


if __name__ == '__main__':
    t0 = time.time(); res = []
    with Pool(4) as pool:
        for i, r in enumerate(pool.imap_unordered(work, cand, chunksize=8)):
            res.append(r)
            if (i + 1) % 500 == 0:
                print(i + 1, time.time() - t0, flush=True)
    with open(base + 'verify_%g.pkl' % TMAX, 'wb') as fh:
        pickle.dump(res, fh)
    print('done', time.time() - t0)
