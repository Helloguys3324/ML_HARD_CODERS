#!/usr/bin/env python
"""Explore grid renderings / orientations / thresholds."""
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
# flat order as in csv/file order
order = pd.read_csv(os.path.join(OUT, 'survey_pred_global.csv'))
F = order['count'].values.reshape(9, 37)

variants = {}
for nm, A in [('pos', G), ('file', F)]:
    for t in (25, 35, 45, 55, 65, 75, 85):
        q = np.percentile(A, t)
        variants[f'{nm}_p{t}'] = (A >= q).astype(np.float32)
    variants[f'{nm}_T'] = A.T
    variants[f'{nm}_T_p65'] = (A.T >= np.percentile(A, 65)).astype(np.float32)

for nm, A in variants.items():
    im = (A * 255).astype(np.uint8)
    sc = max(4, 600 // max(im.shape))
    im = cv2.resize(im, (im.shape[1] * sc, im.shape[0] * sc), interpolation=cv2.INTER_NEAREST)
    im = cv2.copyMakeBorder(im, 20, 4, 4, 4, cv2.BORDER_CONSTANT, value=128)
    cv2.putText(im, nm, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
    cv2.imwrite(os.path.join(OUT, f'var_{nm}.png'), im)
print('saved', len(variants))
