#!/usr/bin/env python
"""Render survey count grid and test encodings."""
import os
import numpy as np
import pandas as pd
import cv2

OUT = '/home/kozak/mllm/MML/work/solutions/task_mongolia'
s = pd.read_csv(os.path.join(OUT, 'survey_plots_ij.csv'))
p = pd.read_csv(os.path.join(OUT, 'survey_pred_global.csv'))
m = dict(zip(p.image, p['count']))
G = np.zeros((9, 37), np.float32)
for _, r in s.iterrows():
    G[int(r.j), int(r.i)] = m[r.image]

print('grid stats', G.min(), G.max(), G.mean(), G.std())
print('row sums', G.sum(1).round(1))
print('col sums', G.sum(0).round(1))
print('row means', G.mean(1).round(1))

# ASCII rendering at several thresholds
for thr in (20, 25, 30, 35, 40, 45, 50):
    print(f'--- thr {thr}')
    for row in G:
        print(''.join('#' if v >= thr else '.' for v in row))

# grayscale render
def render(path, arr, scale=12, cmap_norm=None):
    a = arr.astype(np.float32)
    a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    im = (a * 255).astype(np.uint8)
    im = cv2.resize(im, (im.shape[1] * scale, im.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(path, im)

render(os.path.join(OUT, 'grid_gray.png'), G)
for q in (30, 40, 50, 60, 70):
    render(os.path.join(OUT, 'grid_thr%d.png' % q), (G >= q).astype(np.float32))
print('saved renders')
