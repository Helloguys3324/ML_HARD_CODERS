#!/usr/bin/env python
"""task_mongolia — reproducible driver.

Pipeline
--------
1. Fit the 50 m, 26.6-deg rotated 37x9 plot lattice from survey_plots.csv.
2. Count sheep in survey photos with two from-scratch classical regressors
   (see global_reg.py and cell_density.py + cell_reg_fast.py) and blend them.
3. Write counts.csv.
4. Render the 37x9 census grid (the intended flag canvas).

Usage
-----
    /home/kozak/mllm/py solve.py            # uses cached model predictions
    /home/kozak/mllm/py solve.py --full     # re-runs both counters (~15 min)

The saved prediction files survey_pred_global.csv / survey_pred_cell.csv were
produced by:
    /home/kozak/mllm/py global_reg.py train     # 46 global features, 5-fold CV
    /home/kozak/mllm/py global_reg.py survey
    /home/kozak/mllm/py cell_density.py train   # 60x60 cell features
    /home/kozak/mllm/py cell_density.py survey
    /home/kozak/mllm/py cell_reg_fast.py        # LightGBM per-cell regressor
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import cv2
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter

DATA = '/home/kozak/mllm/MML/work/tasks/task_mongolia/data'
OUT = os.path.dirname(os.path.abspath(__file__))
PY = '/home/kozak/mllm/py'


def fit_lattice(csv_path):
    """Recover integer (i, j) lattice coordinates for every plot.

    Basis vectors are estimated from the mean of the nearest-neighbour
    displacement clusters, then points are projected on the reciprocal basis.
    Returns an (N,2) integer array and the basis matrix.
    """
    s = pd.read_csv(csv_path)
    P = s[['x_m', 'y_m']].values.astype(np.float64)
    t = cKDTree(P)
    pairs = t.query_pairs(r=52, output_type='ndarray')
    D = P[pairs[:, 1]] - P[pairs[:, 0]]
    D[D[:, 0] < 0] *= -1  # canonical orientation

    def cluster_mean(cand, tol=3.0):
        m = np.linalg.norm(D - np.array(cand), axis=1) < tol
        return D[m].mean(0)

    u = cluster_mean((45.0, 22.4))
    v = cluster_mean((22.4, -44.2))
    M = np.column_stack([u, v])
    A = (P - P[0]) @ np.linalg.inv(M).T
    assert np.abs(A - np.round(A)).max() < 0.5, 'lattice fit failed'
    I = np.round(A).astype(int)
    I -= I.min(0)
    assert len(np.unique(I[:, 0])) * len(np.unique(I[:, 1])) == len(I)
    return s, I, M


def render_grid(ij, counts_by_image, path):
    G = np.zeros((9, 37), np.float32)
    for image, i, j in zip(ij.image, ij.i, ij.j):
        G[int(j), int(i)] = counts_by_image[image]
    z = (G - G.min()) / (G.max() - G.min() + 1e-9)
    im = (z * 255).astype(np.uint8)
    im = cv2.resize(im, (37 * 24, 9 * 24), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(path, im)
    # debiased binary view (row/column background removed)
    R = G - gaussian_filter(G, 2.0)
    R = (R - R.mean()) / (R.std() + 1e-9)
    cv2.imwrite(path.replace('.png', '_resid.png'),
                cv2.resize(((R > 0.35) * 255).astype(np.uint8),
                           (37 * 24, 9 * 24), interpolation=cv2.INTER_NEAREST))
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', action='store_true', help='re-run both counters')
    args = ap.parse_args()

    if args.full:
        for cmd in (['global_reg.py', 'train'], ['global_reg.py', 'survey'],
                    ['cell_density.py', 'train'], ['cell_density.py', 'survey'],
                    ['cell_reg_fast.py']):
            subprocess.run([PY, '-u'] + cmd, cwd=OUT, check=True)

    s, I, M = fit_lattice(os.path.join(DATA, 'survey_plots.csv'))
    s['i'] = I[:, 0]
    s['j'] = I[:, 1]
    s.to_csv(os.path.join(OUT, 'survey_plots_ij.csv'), index=False)
    print('lattice basis\n', M)
    print('grid', I[:, 0].max() + 1, 'x', I[:, 1].max() + 1)

    g = pd.read_csv(os.path.join(OUT, 'survey_pred_global.csv')).set_index('image')['count']
    c = pd.read_csv(os.path.join(OUT, 'survey_pred_cell.csv')).set_index('image')['count']
    names = sorted(g.index)
    counts = np.round((0.5 * g.loc[names] + 0.5 * c.loc[names]).clip(lower=0)).astype(int)
    pd.DataFrame({'image': names, 'count': counts}).to_csv(
        os.path.join(OUT, 'counts.csv'), index=False)
    print('counts written:', counts.sum(), 'sheep total;',
          'mean', round(float(counts.mean()), 1))

    G = render_grid(s, dict(zip(names, counts)), os.path.join(OUT, 'census_grid.png'))
    print('\n37x9 census grid (estimated counts):')
    np.set_printoptions(linewidth=250)
    print(np.round(G).astype(int))


if __name__ == '__main__':
    main()
