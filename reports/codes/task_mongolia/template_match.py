#!/usr/bin/env python
"""Template matching + peak counting, calibrated against point annotations."""
import os, time, json, sys
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import maximum_filter

DATA = '/home/kozak/mllm/MML/work/tasks/task_mongolia/data'
OUT = '/home/kozak/mllm/MML/work/solutions/task_mongolia'
cv2.setNumThreads(2)


def whiteness(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[:, :, 1] / 255.0
    v = hsv[:, :, 2] / 255.0
    return (v * (1.0 - s) ** 1.2).astype(np.float32)


def build_templates(pts, names, k=40, half=14, n_per=15, rng=None):
    patches = []
    for n in names[:k]:
        im = cv2.imread(os.path.join(DATA, 'train', n))
        w = whiteness(im)
        P = pts[pts.image == n][['x', 'y']].values
        if len(P) == 0:
            continue
        sel = P[rng.choice(len(P), min(n_per, len(P)), replace=False)]
        for x, y in sel:
            xi, yi = int(round(x)), int(round(y))
            if half <= xi < w.shape[1] - half and half <= yi < w.shape[0] - half:
                p = w[yi - half:yi + half + 1, xi - half:xi + half + 1].copy()
                p = (p - p.mean()) / (p.std() + 1e-6)
                patches.append(p)
    patches = np.array(patches)
    tpl = patches.mean(0)
    tpl = (tpl - tpl.mean()) / (tpl.std() + 1e-6)
    return tpl.astype(np.float32), patches


def response_map(w, tpl):
    r = cv2.matchTemplate(w, tpl, cv2.TM_CCOEFF_NORMED)
    pad = tpl.shape[0] // 2
    return cv2.copyMakeBorder(r, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)


def peaks(resp, md=13, thr=0.35):
    mx = maximum_filter(resp, size=md)
    m = (resp >= mx - 1e-7) & (resp >= thr)
    ys, xs = np.nonzero(m)
    return np.c_[xs, ys], resp[ys, xs]


def main():
    counts = pd.read_csv(os.path.join(DATA, 'train_counts.csv')).set_index('image')['count']
    pts = pd.read_csv(os.path.join(DATA, 'train_points.csv'))
    names = list(counts.index)
    rng = np.random.default_rng(0)
    rng.shuffle(names)
    tpl, patches = build_templates(pts, names[:40], rng=rng)
    print('template built', tpl.shape, 'n patches', len(patches), flush=True)
    # save template visualization
    tv = ((tpl - tpl.min()) / (tpl.max() - tpl.min()) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUT, 'template.png'), cv2.resize(tv, (280, 280), interpolation=cv2.INTER_NEAREST))

    evalset = names[40:160]
    thr_list = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    stats = {t: {'tp': 0, 'fp': 0, 'fn': 0} for t in thr_list}
    pred_counts = {t: [] for t in thr_list}
    t0 = time.time()
    for k, n in enumerate(evalset):
        im = cv2.imread(os.path.join(DATA, 'train', n))
        w = whiteness(im)
        resp = response_map(w, tpl)
        P = pts[pts.image == n][['x', 'y']].values
        for t in thr_list:
            pk, h = peaks(resp, md=13, thr=t)
            pred_counts[t].append(len(pk))
            # match to GT
            if len(P) == 0:
                stats[t]['fp'] += len(pk)
                continue
            from scipy.spatial import cKDTree
            if len(pk):
                tr = cKDTree(P)
                d, _ = tr.query(pk, k=1)
                tp = (d <= 12).sum()
                stats[t]['tp'] += tp
                stats[t]['fp'] += len(pk) - tp
            stats[t]['fn'] += len(P) - 0 if False else 0
            tr2 = cKDTree(pk) if len(pk) else None
            if tr2 is not None:
                dg, _ = tr2.query(P, k=1)
                stats[t]['fn'] += (dg > 12).sum()
            else:
                stats[t]['fn'] += len(P)
        if k % 20 == 0:
            print(k, 't', round(time.time() - t0), flush=True)
    truth = np.array([counts[n] for n in evalset], float)
    for t in thr_list:
        s = stats[t]
        prec = s['tp'] / max(1, s['tp'] + s['fp'])
        rec = s['tp'] / max(1, s['tp'] + s['fn'])
        pred = np.array(pred_counts[t], float)
        mae = np.abs(pred - truth).mean()
        print(f'thr={t:.2f} prec={prec:.3f} rec={rec:.3f} F1={2*prec*rec/max(1e-9,prec+rec):.3f} '
              f'MAE={mae:.2f} corr={np.corrcoef(pred,truth)[0,1]:.3f}', flush=True)
    # save
    np.save(os.path.join(OUT, 'evalset_truth.npy'), truth)
    json.dump({'names': evalset}, open(os.path.join(OUT, 'evalset_names.json'), 'w'))
    for t in thr_list:
        np.save(os.path.join(OUT, f'evalset_pred_{t}.npy'), np.array(pred_counts[t]))
    np.save(os.path.join(OUT, 'template.npy'), tpl)


if __name__ == '__main__':
    main()
