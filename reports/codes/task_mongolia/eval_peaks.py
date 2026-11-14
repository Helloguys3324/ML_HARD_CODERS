#!/usr/bin/env python
"""Tune peak detection on validation probability maps and apply to survey."""
import os, json, sys, itertools
import numpy as np
import pandas as pd
import cv2
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter

DATA = '/home/kozak/mllm/MML/work/tasks/task_mongolia/data'
OUT = '/home/kozak/mllm/MML/work/solutions/task_mongolia'


def count_peaks(p, thr, min_dist, sigma=2.0):
    ps = gaussian_filter(p.astype(np.float32), sigma)
    coords = peak_local_max(ps, min_distance=min_dist, threshold_abs=thr, exclude_border=False)
    return len(coords), coords


def main():
    val = np.load(os.path.join(OUT, 'val_probs.npy'))
    names = json.load(open(os.path.join(OUT, 'val_prob_names.json')))['names']
    counts = pd.read_csv(os.path.join(DATA, 'train_counts.csv')).set_index('image')
    truth = np.array([counts.loc[n, 'count'] for n in names], float)
    print('val n', len(names), 'truth mean', truth.mean())

    # integral regression: count ~ a * sum(p) + b
    sums = val.sum((1, 2), dtype=np.float64)
    A = np.c_[sums, np.ones_like(sums)]
    coef, *_ = np.linalg.lstsq(A, truth, rcond=None)
    pred_int = A @ coef
    print('integral model coef', coef, 'MAE', np.abs(pred_int - truth).mean(),
          'corr', np.corrcoef(pred_int, truth)[0, 1])
    for cap in (0.02, 0.05, 0.1):
        s2 = np.maximum(sums - cap * val.shape[1] * val.shape[2], 0)
        A2 = np.c_[s2, np.ones_like(s2)]
        c2, *_ = np.linalg.lstsq(A2, truth, rcond=None)
        p2 = A2 @ c2
        print(f'  capped integral cap={cap} coef={c2} MAE={np.abs(p2-truth).mean():.3f}')

    best = None
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
        for md in (5, 7, 9, 11):
            preds = np.array([count_peaks(p, thr, md)[0] for p in val], float)
            mae = np.abs(preds - truth).mean()
            # binary presence accuracy
            acc = ((preds > 0) == (truth > 0)).mean()
            if best is None or mae < best[0]:
                best = (mae, thr, md, acc, preds)
            print(f'thr={thr} md={md} MAE={mae:.2f} presence_acc={acc:.3f}')
    mae, thr, md, acc, preds = best
    print('BEST', best[:4])
    print('binary presence errors:', [(names[i], int(truth[i]), int(preds[i])) for i in range(len(names))
                                      if (preds[i] > 0) != (truth[i] > 0)])
    json.dump({'thr': thr, 'min_dist': md, 'integral_coef': coef.tolist()},
              open(os.path.join(OUT, 'peak_params.json'), 'w'))


if __name__ == '__main__':
    main()
