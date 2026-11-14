#!/usr/bin/env python
"""Multi-template bank detector: max NCC over real sheep patches, peak counting."""
import os, sys, time
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import maximum_filter

DATA = '/home/kozak/mllm/MML/work/tasks/task_mongolia/data'
OUT = '/home/kozak/mllm/MML/work/solutions/task_mongolia'
cv2.setNumThreads(2)
SC = 0.5
PS = 19
HALF = PS // 2


def white(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[:, :, 1] / 255.0
    v = hsv[:, :, 2] / 255.0
    return (v * (1.0 - s) ** 1.2).astype(np.float32)


def load(path):
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, None, fx=SC, fy=SC, interpolation=cv2.INTER_AREA)
    return im


def build_bank(pts, names, n_img=60, per=3, rng=None):
    bank = []
    for n in names[:n_img]:
        im = load(os.path.join(DATA, 'train', n))
        w = white(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        P = pts[pts.image == n][['x', 'y']].values * SC
        if len(P) == 0:
            continue
        idx = rng.choice(len(P), min(per, len(P)), replace=False)
        for x, y in P[idx]:
            xi, yi = int(round(x)), int(round(y))
            if HALF <= xi < w.shape[1] - HALF and HALF <= yi < w.shape[0] - HALF:
                p = w[yi - HALF:yi + HALF + 1, xi - HALF:xi + HALF + 1].copy()
                p = (p - p.mean()) / (p.std() + 1e-6)
                bank.append(p)
    return np.array(bank, np.float32)


def response(w, bank, stride=2):
    resp = None
    for t in bank[::stride]:
        r = cv2.matchTemplate(w, t, cv2.TM_CCOEFF_NORMED)
        if resp is None:
            resp = r
        else:
            np.maximum(resp, r, out=resp)
    pad = HALF
    return cv2.copyMakeBorder(resp, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)


def count(im, bank, thr=0.45, md=7):
    w = white(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    r = response(w, bank)
    mx = maximum_filter(r, size=md)
    m = (r >= mx - 1e-9) & (r > thr)
    return int(m.sum())


if __name__ == '__main__':
    counts = pd.read_csv(os.path.join(DATA, 'train_counts.csv')).set_index('image')['count']
    pts = pd.read_csv(os.path.join(DATA, 'train_points.csv'))
    names = list(counts.index)
    rng = np.random.default_rng(3)
    rng.shuffle(names)
    bank = build_bank(pts, names[:60], per=2, rng=rng)
    print('bank', bank.shape, flush=True)
    evalnames = names[60:110]
    t0 = time.time()
    res = {}
    for k, n in enumerate(evalnames):
        im = load(os.path.join(DATA, 'train', n))
        w = white(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        r = response(w, bank)
        res[n] = r
        if k % 10 == 0:
            print(k, round(time.time() - t0), flush=True)
    truth = np.array([counts[n] for n in evalnames], float)
    for thr in (0.35, 0.4, 0.45, 0.5, 0.55, 0.6):
        for md in (5, 7, 9):
            preds = []
            for n in evalnames:
                r = res[n]
                mx = maximum_filter(r, size=md)
                preds.append(int(((r >= mx - 1e-9) & (r > thr)).sum()))
            preds = np.array(preds, float)
            mae = np.abs(preds - truth).mean()
            print(f'thr={thr} md={md} MAE={mae:.2f} corr={np.corrcoef(preds,truth)[0,1]:.3f} '
                  f'presAcc={((preds>0)==(truth>0)).mean():.3f}', flush=True)
    np.save(os.path.join(OUT, 'tplbank.npy'), bank)
