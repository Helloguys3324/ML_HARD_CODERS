#!/usr/bin/env python
"""Fast blob-count baseline using DoG local maxima; calibrate count vs truth."""
import os, sys, time, json
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import maximum_filter, gaussian_filter

DATA = '/home/kozak/mllm/MML/work/tasks/task_mongolia/data'
OUT = '/home/kozak/mllm/MML/work/solutions/task_mongolia'
cv2.setNumThreads(2)


def score(img_bgr):
    img = img_bgr.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[:, :, 1] / 255.0
    v = hsv[:, :, 2] / 255.0
    white = v * (1.0 - s) ** 1.2
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] / 255.0
    # multi-scale DoG on whiteness
    d1 = cv2.GaussianBlur(white, (0, 0), 3) - cv2.GaussianBlur(white, (0, 0), 9)
    d2 = cv2.GaussianBlur(white, (0, 0), 5) - cv2.GaussianBlur(white, (0, 0), 14)
    sc = np.maximum(d1, d2)
    # dark sheep (black top-hat on L)
    L8 = np.clip(L * 255, 0, 255).astype(np.uint8)
    bth = cv2.morphologyEx(L8, cv2.MORPH_BLACKHAT,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))).astype(np.float32) / 255.0
    bth = cv2.GaussianBlur(bth, (0, 0), 2)
    return sc, bth


def peak_heights(sc, md=13):
    mx = maximum_filter(sc, size=md)
    peaks = (sc >= mx - 1e-9)
    ys, xs = np.nonzero(peaks)
    return sc[ys, xs]


def main():
    counts = pd.read_csv(os.path.join(DATA, 'train_counts.csv')).set_index('image')['count']
    names = list(counts.index)
    rng = np.random.default_rng(1)
    rng.shuffle(names)
    subset = names[:120]
    thrs = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20])
    rows = []
    t0 = time.time()
    for k, n in enumerate(subset):
        im = cv2.imread(os.path.join(DATA, 'train', n))
        sc, bth = score(im)
        h = peak_heights(sc)
        hb = peak_heights(bth)
        rows.append((n, counts[n], h, hb))
        if k % 20 == 0:
            print(k, 't', round(time.time() - t0), flush=True)
    # evaluate thresholds after collecting peaks
    for use_dark in (False, True):
        for thr in thrs:
            preds = []
            for n, c, h, hb in rows:
                p = (h >= thr).sum()
                if use_dark:
                    p += (hb >= thr).sum()
                preds.append(p)
            preds = np.array(preds, float)
            truth = np.array([r[1] for r in rows], float)
            mae = np.abs(preds - truth).mean()
            acc = ((preds > 0) == (truth > 0)).mean()
            print(f'dark={use_dark} thr={thr:.2f} MAE={mae:.2f} corr={np.corrcoef(preds,truth)[0,1]:.3f} presence={acc:.3f}', flush=True)
    # save peak data for calibration
    np.save(os.path.join(OUT, 'peak_rows.npy'), np.array(rows, dtype=object), allow_pickle=True)


if __name__ == '__main__':
    main()
