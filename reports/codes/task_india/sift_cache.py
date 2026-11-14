#!/usr/bin/env python3
"""Compute & cache SIFT keypoints/descriptors for all images (dev helper)."""
import os, sys, glob, pickle, numpy as np, cv2
from multiprocessing import Pool

cv2.setNumThreads(1)
MAXDIM = 900


def one(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    if max(h, w) > MAXDIM:
        s = MAXDIM / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    sift = cv2.SIFT_create(nfeatures=600, contrastThreshold=0.01)
    kp, des = sift.detectAndCompute(img, None)
    if des is None:
        des = np.zeros((0, 128), np.float32)
        pts = np.zeros((0, 2), np.float32)
    else:
        pts = np.float32([k.pt for k in kp])
    return os.path.basename(path), img.shape, pts, des


if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else '/home/kozak/mllm/MML/work/tasks/task_india/data/train'
    out = sys.argv[2] if len(sys.argv) > 2 else '/home/kozak/mllm/MML/work/solutions/task_india/sift.pkl'
    paths = sorted(glob.glob(os.path.join(root, '*.jpg')) + glob.glob(os.path.join(root, '*.png')))
    print(len(paths), 'images', flush=True)
    res = {}
    with Pool(4) as pool:
        for i, r in enumerate(pool.imap_unordered(one, paths, chunksize=4)):
            if r is not None:
                res[r[0]] = (r[1], r[2], r[3])
            if (i + 1) % 200 == 0:
                print(i + 1, flush=True)
    with open(out, 'wb') as fh:
        pickle.dump(res, fh, protocol=4)
    print('saved', len(res))
