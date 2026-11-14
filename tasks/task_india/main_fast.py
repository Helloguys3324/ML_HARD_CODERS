#!/usr/bin/env python3
"""
Twins Separated at Birth -- duplicate-pair detection for Indian monument photos.

Pipeline
--------
1.  Discover every image in the provided archive (zip or directory tree).
2.  Multi-zoom perceptual hashing (DCT pHash, 8 central-crop zoom levels).
    Duplicates in this dataset are produced by zooming/cropping one source
    photograph and re-encoding it, so a pair matches when the full view of one
    image matches a zoomed view of the other.
3.  Candidate pairs = (aspect-ratio ratio >= AR_T) AND (directed zoom-hash
    distance <= HASH_T).  The aspect ratio of duplicates is preserved by the
    generator, which is a very cheap and safe filter.
4.  Geometric verification of every candidate with ORB keypoints +
    RANSAC similarity transform, plus normalized cross-correlation of the
    warped overlap.  True duplicates have hundreds of geometrically consistent
    matches and near-perfect correlation; different photographs of the same
    monument do not.
5.  Write output.csv with the columns image_a,image_b.

The program is self-contained (numpy + OpenCV only) and reads its input either
from a directory, a .zip archive, or an archive located in the working
directory.
"""
from __future__ import annotations

import csv
import os
import sys
import glob
import math
import time
import zipfile
import tempfile
import numpy as np
import cv2

cv2.setNumThreads(1)

# ----------------------------- configuration --------------------------------
FRACS = [1.0, 0.9, 0.84, 0.78, 0.72, 0.66, 0.60, 0.54]  # central crop factors
HASH_T = 16          # directed multi-zoom pHash hamming threshold for candidates
AR_T = 0.97          # min aspect-ratio ratio for a candidate pair
ORB_FEATURES = 300   # keypoints per image for verification
ORB_MAXDIM = 384     # images resized so max(side) <= this for verification
RATIO_TEST = 0.75    # Lowe ratio for ORB matching
MIN_INLIERS = 40     # minimum RANSAC inliers to accept a pair
MIN_NCC = 0.70       # minimum correlation of warped overlap to accept
MIN_STRONG_HASH = 8  # hash distance that counts as a "strong" match
MIN_STRONG_INLIERS = 20  # ... but only when the keypoints back it up
MIN_INLIER_FRAC = 0.60  # minimum fraction of ratio-test matches that are inliers
STRONG_AR = 0.985    # aspect ratio threshold for the strong-hash shortcut
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')


# ----------------------------- input discovery ------------------------------
def _images_in(directory):
    out = []
    for root, _dirs, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                out.append(os.path.join(root, f))
    return sorted(out)


def find_images(base=None):
    """Locate all images, extracting archives if needed."""
    here = os.path.abspath(base or os.getcwd())
    if os.path.isfile(here):
        paths = _images_in(os.path.dirname(here))
        if paths:
            return paths
        zips = [here] if here.lower().endswith('.zip') else []
        zips += sorted(glob.glob(os.path.join(os.path.dirname(here), '*.zip')))
    else:
        paths = _images_in(here)
        if paths:
            return paths
        zips = sorted(glob.glob(os.path.join(here, '*.zip')))
    # Nothing loose -> look for an archive beside the script / in a parent dir.
    zips += sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), '*.zip')))
    if not zips:
        parent = os.path.dirname(here if os.path.isdir(here) else os.path.dirname(here))
        zips += sorted(glob.glob(os.path.join(parent, '*.zip')))
    for z in zips:
        try:
            tmp = tempfile.mkdtemp(prefix='twins_')
            with zipfile.ZipFile(z) as zf:
                zf.extractall(tmp)
            paths = _images_in(tmp)
            if paths:
                return paths
        except Exception:
            continue
    return []


# ----------------------------- perceptual hash ------------------------------
def _phash_gray(img, ds=32, hs=8):
    img = cv2.resize(img, (ds, ds), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(img)
    low = d[:hs, :hs].copy()
    med = np.median(low[1:, :])
    bits = (low > med).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return np.uint64(val)


def hash_one(path):
    img = cv2.imread(path, cv2.IMREAD_REDUCED_GRAYSCALE_2)
    if img is None:
        return None
    h, w = img.shape[:2]
    if h < 8 or w < 8:
        return None
    g = img
    hs = []
    for f in FRACS:
        ch, cw = max(8, int(round(h * f))), max(8, int(round(w * f)))
        ch, cw = min(ch, h), min(cw, w)
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        hs.append(_phash_gray(g[y0:y0 + ch, x0:x0 + cw]))
    return np.array(hs, dtype=np.uint64), h, w


def _popcount(x):
    x = x.astype(np.uint64)
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int16)


# parallel helpers need module-level state
_HASHES = None
_DIMS = None


def _hash_worker_init(hashes, dims):
    global _HASHES, _DIMS
    _HASHES = hashes
    _DIMS = dims


def _pair_block(args):
    """For image indices [r0:r1], return candidate pairs (i, j, dist) with j > i."""
    r0, r1 = args
    H = _HASHES
    n = H.shape[0]
    L = H.shape[1]
    best = np.full((r1 - r0, n), 64, dtype=np.int16)
    # directed comparisons: full view vs zoomed view, both orderings
    for j in range(L):
        sub = H[r0:r1, 0][:, None]
        np.minimum(best, _popcount(sub ^ H[:, j][None, :]), out=best)
        if j > 0:
            sub2 = H[r0:r1, j][:, None]
            np.minimum(best, _popcount(sub2 ^ H[:, 0][None, :]), out=best)
    dims = _DIMS
    ar = dims[:, 0].astype(np.float32) / np.maximum(dims[:, 1], 1)
    out = []
    for i in range(r1 - r0):
        gi = r0 + i
        row = best[i]
        cols = np.nonzero(row <= HASH_T)[0]
        cols = cols[cols > gi]
        if len(cols) == 0:
            continue
        arj = ar[cols]
        ok = (np.minimum(ar[gi], arj) / np.maximum(ar[gi], arj)) >= AR_T
        for c, d in zip(cols[ok], row[cols[ok]]):
            out.append((gi, int(c), int(d)))
    return out


def candidate_pairs(hashes, dims, workers):
    from multiprocessing import Pool
    n = len(hashes)
    step = 256 if n > 6000 else 384
    blocks = [(s, min(s + step, n)) for s in range(0, n, step)]
    deltas = []
    if workers > 1:
        with Pool(workers, initializer=_hash_worker_init, initargs=(hashes, dims)) as pool:
            for res in pool.imap_unordered(_pair_block, blocks, chunksize=1):
                deltas.extend(res)
    else:
        _hash_worker_init(hashes, dims)
        for b in blocks:
            deltas.extend(_pair_block(b))
    return deltas


# ----------------------------- ORB verification -----------------------------
_ORB = None


def orb_feature(path):
    img = cv2.imread(path, cv2.IMREAD_REDUCED_GRAYSCALE_2)
    if img is None:
        return None
    h, w = img.shape
    if max(h, w) > ORB_MAXDIM:
        s = ORB_MAXDIM / float(max(h, w))
        img = cv2.resize(img, (max(8, int(w * s)), max(8, int(h * s))),
                         interpolation=cv2.INTER_AREA)
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES, fastThreshold=10)
    kp, des = orb.detectAndCompute(img, None)
    pts = np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32)
    if des is None:
        des = np.zeros((0, 32), np.uint8)
    return img, pts, des


def verify_pair(args):
    a, b, hd = args
    fa = _ORB.get(a)
    fb = _ORB.get(b)
    if fa is None or fb is None:
        return (a, b, hd, 0, 0.0, 0.0, 0.0)
    ia, pa, da = fa
    ib, pb, db = fb
    if len(da) < 6 or len(db) < 6:
        return (a, b, hd, 0, 0.0, 0.0, 0.0)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    m = bf.knnMatch(da, db, k=2)
    good = [x for x, y in m if x.distance < RATIO_TEST * y.distance]
    if len(good) < 6:
        return (a, b, hd, len(good), 0.0, 0.0, 0.0)
    src = np.float32([pa[g.queryIdx] for g in good]).reshape(-1, 1, 2)
    dst = np.float32([pb[g.trainIdx] for g in good]).reshape(-1, 1, 2)
    A, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                          ransacReprojThreshold=3.0, maxIters=2000)
    if A is None or mask is None:
        return (a, b, hd, len(good), 0.0, 0.0, 0.0)
    inl = int(mask.sum())
    frac = inl / max(len(good), 1)
    s = float(math.hypot(A[0, 0], A[1, 0]))
    hb, wb = ib.shape
    warp = cv2.warpAffine(ia, A, (wb, hb), flags=cv2.INTER_LINEAR, borderValue=0)
    msk = cv2.warpAffine(np.ones_like(ia), A, (wb, hb), flags=cv2.INTER_NEAREST, borderValue=0) > 0
    # require a meaningful overlap in both images
    ov = msk.sum() / float(hb * wb)
    if msk.sum() < 200 or ov < 0.12:
        ncc = 0.0
    else:
        va = warp[msk].astype(np.float32)
        vb = ib[msk].astype(np.float32)
        va -= va.mean()
        vb -= vb.mean()
        ncc = float((va * vb).sum() / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-6))
    return (a, b, hd, inl, frac, s, ncc)


# ----------------------------- main ----------------------------------------
def main(argv):
    t_start = time.time()
    debug_path = os.environ.get('TWINS_DEBUG')
    input_dir = None
    if len(argv) > 1 and not argv[1].startswith('-'):
        input_dir = argv[1]
    paths = find_images(input_dir)
    if not paths:
        print('ERROR: no input images found', file=sys.stderr)
        with open('output.csv', 'w', newline='') as fh:
            csv.writer(fh).writerow(['image_a', 'image_b'])
        return 1
    print(f'[twins] {len(paths)} images found', flush=True)

    # Keep basenames; if duplicates in basenames, disambiguate with relpath.
    names = [os.path.basename(p) for p in paths]
    if len(set(names)) != len(names):
        stems = [os.path.splitext(os.path.relpath(p, input_dir or os.getcwd()))[0]
                 for p in paths]
        names = [s.replace(os.sep, '_') + os.path.splitext(p)[1] for s, p in zip(stems, paths)]
    n = len(paths)

    workers = max(1, min(4, os.cpu_count() or 1))

    # ---- 1. hashes ----
    from multiprocessing import Pool
    hashes = np.zeros((n, len(FRACS)), dtype=np.uint64)
    dims = np.ones((n, 2), dtype=np.int32)
    ok = np.zeros(n, dtype=bool)
    t = time.time()
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap(hash_one, paths, chunksize=16)):
            if r is not None:
                hashes[i] = r[0]
                dims[i] = r[1], r[2]
                ok[i] = True
    print(f'[twins] hashes done {time.time()-t:.1f}s', flush=True)
    if ok.sum() < n:
        keep = np.nonzero(ok)[0]
        paths = [paths[i] for i in keep]
        names = [names[i] for i in keep]
        hashes = hashes[keep]
        dims = dims[keep]
        n = len(paths)

    # ---- 2. candidate pairs ----
    t = time.time()
    cand = candidate_pairs(hashes, dims, workers)
    print(f'[twins] {len(cand)} candidate pairs ({time.time()-t:.1f}s)', flush=True)
    # Safety valve for unexpectedly large archives: keep the most similar pairs.
    if len(cand) > 140000:
        cand.sort(key=lambda t: t[2])
        cand = cand[:140000]
        print(f'[twins] capped to {len(cand)} candidates', flush=True)

    # ---- 3. ORB verification ----
    involved = sorted({i for a, b, d in cand for i in (a, b)})
    t = time.time()
    feats = {}
    with Pool(workers) as pool:
        for i, r in zip(involved, pool.imap(orb_feature, [paths[i] for i in involved],
                                           chunksize=4)):
            feats[i] = r
    print(f'[twins] ORB features done {time.time()-t:.1f}s for {len(involved)} images',
          flush=True)

    global _ORB
    _ORB = feats
    t = time.time()
    accepted = []
    records = []
    with Pool(workers) as pool:
        for r in pool.imap_unordered(verify_pair, cand, chunksize=16):
            a, b, hd, inl, frac, s, ncc = r
            ar_a = dims[a, 0] / max(dims[a, 1], 1)
            ar_b = dims[b, 0] / max(dims[b, 1], 1)
            ar_ratio = min(ar_a, ar_b) / max(ar_a, ar_b)
            # scale sanity: duplicates are zoom/crop pairs, never extreme warps
            scale_ok = (0.35 <= s <= 1.45) or (s <= 0.05 and inl >= MIN_INLIERS)
            strong = (hd <= MIN_STRONG_HASH and ar_ratio >= STRONG_AR
                      and inl >= MIN_STRONG_INLIERS)
            geometric = (inl >= MIN_INLIERS and frac >= MIN_INLIER_FRAC
                         and ncc >= MIN_NCC and scale_ok)
            if debug_path:
                records.append((a, b, hd, inl, frac, s, ncc))
            if geometric or strong:
                accepted.append((a, b))
    print(f'[twins] verification done {time.time()-t:.1f}s, {len(accepted)} pairs',
          flush=True)

    if debug_path:
        import pickle
        with open(debug_path, 'wb') as fh:
            pickle.dump({'records': records, 'names': names,
                         'dims': dims}, fh, protocol=4)
        print(f'[twins] debug records -> {debug_path}', flush=True)

    accepted.sort()
    with open('output.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['image_a', 'image_b'])
        for a, b in accepted:
            w.writerow([names[a], names[b]])
    print(f'[twins] wrote output.csv with {len(accepted)} pairs '
          f'({time.time()-t_start:.1f}s total)', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
