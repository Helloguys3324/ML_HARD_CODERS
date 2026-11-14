#!/usr/bin/env python
"""Cell-level density regression: partition image into cells, regress sheep count per cell."""
import os, sys, time, json
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import maximum_filter

DATA = '/home/kozak/mllm/MML/work/tasks/task_mongolia/data'
OUT = '/home/kozak/mllm/MML/work/solutions/task_mongolia'
cv2.setNumThreads(2)
CS = 60  # cell size px -> 10x10 grid


def maps(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[:, :, 1] / 255.0
    v = hsv[:, :, 2] / 255.0
    white = v * (1.0 - s) ** 1.2
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] / 255.0
    w8 = np.clip(white * 255, 0, 255).astype(np.uint8)
    th15 = cv2.morphologyEx(w8, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))).astype(np.float32) / 255
    th23 = cv2.morphologyEx(w8, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))).astype(np.float32) / 255
    d1 = cv2.GaussianBlur(white, (0, 0), 3) - cv2.GaussianBlur(white, (0, 0), 9)
    d2 = cv2.GaussianBlur(white, (0, 0), 5) - cv2.GaussianBlur(white, (0, 0), 14)
    sc = np.maximum(d1, d2)
    L8 = np.clip(L * 255, 0, 255).astype(np.uint8)
    bth = cv2.morphologyEx(L8, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))).astype(np.float32) / 255
    mx = maximum_filter(sc, size=13)
    peak = (sc >= mx - 1e-9)
    return white, th15, th23, sc, peak, bth, L, s, v


def cell_feats(img_bgr, pts=None):
    white, th15, th23, sc, peak, bth, L, s, v = maps(img_bgr)
    H, W = white.shape
    ny, nx = H // CS, W // CS
    F = []
    for cy in range(ny):
        for cx in range(nx):
            sl = np.s_[cy * CS:(cy + 1) * CS, cx * CS:(cx + 1) * CS]
            w = white[sl]; t15 = th15[sl]; t23 = th23[sl]; b = bth[sl]
            p = peak[sl]; sv = sc[sl]; l = L[sl]; ss = s[sl]; vv = v[sl]
            ph = sv[p]
            f = [w.mean(), w.std(), (w > 0.5).mean(), (w > 0.6).mean(), (w > 0.7).mean(),
                 t15.mean(), (t15 > 0.2).mean(), (t15 > 0.3).mean(),
                 t23.mean(), (t23 > 0.2).mean(),
                 float((ph > 0.15).sum()), float((ph > 0.2).sum()), float((ph > 0.25).sum()),
                 ph.max() if len(ph) else 0.0, ph.mean() if len(ph) else 0.0,
                 b.mean(), (b > 0.25).mean(),
                 l.mean(), l.std(), ss.mean(), vv.mean()]
            F.append(f)
    F = np.array(F, np.float32)
    if pts is not None:
        cnt = np.zeros((ny, nx), np.float32)
        for x, y in pts:
            ci, cj = int(y) // CS, int(x) // CS
            if 0 <= ci < ny and 0 <= cj < nx:
                cnt[ci, cj] += 1
        return F, cnt.ravel()
    return F


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'train'
    if mode == 'train':
        counts = pd.read_csv(os.path.join(DATA, 'train_counts.csv'))
        pts = pd.read_csv(os.path.join(DATA, 'train_points.csv'))
        pts_by = {n: g[['x', 'y']].values for n, g in pts.groupby('image')}
        names = counts.image.tolist()
        X, Y, imgids = [], [], []
        t0 = time.time()
        for k, n in enumerate(names):
            im = cv2.imread(os.path.join(DATA, 'train', n))
            F, c = cell_feats(im, pts_by.get(n, np.zeros((0, 2))))
            X.append(F); Y.append(c); imgids += [k] * len(c)
            if k % 40 == 0:
                print(k, 't', round(time.time() - t0), flush=True)
        X = np.concatenate(X); Y = np.concatenate(Y); imgids = np.array(imgids)
        np.savez(os.path.join(OUT, 'cell_train.npz'), X=X, Y=Y, imgids=imgids,
                 imgcounts=counts['count'].values.astype(np.float32),
                 names=np.array(names))
        print('saved', X.shape, flush=True)
    elif mode == 'survey':
        sdir = os.path.join(DATA, 'survey')
        snames = sorted(os.listdir(sdir))
        X, imgids = [], []
        t0 = time.time()
        for k, n in enumerate(snames):
            im = cv2.imread(os.path.join(sdir, n))
            F = cell_feats(im)
            X.append(F); imgids += [k] * len(F)
            if k % 40 == 0:
                print(k, 't', round(time.time() - t0), flush=True)
        np.savez(os.path.join(OUT, 'cell_survey.npz'), X=np.concatenate(X),
                 imgids=np.array(imgids), names=np.array(snames))
        print('saved survey', flush=True)


if __name__ == '__main__':
    main()
