#!/usr/bin/env python
"""Produce final counts.csv by blending global + cell-level LightGBM regressors.

Reproducible pipeline summary (see solution.md):
 1. survey_plots.csv -> lattice fit -> 37x9 plot grid (rotated 26.6 deg, 50 m pitch)
 2. 280 train photos with exact counts + point annotations
 3. Two classical regressors (global image features and 60x60 cell features)
 4. Blend of both on the 333 survey photos -> counts.csv
"""
import numpy as np
import pandas as pd

DATA = '/home/kozak/mllm/MML/work/tasks/task_mongolia/data'
OUT = '/home/kozak/mllm/MML/work/solutions/task_mongolia'

g = pd.read_csv(f'{OUT}/survey_pred_global.csv').set_index('image')['count']
c = pd.read_csv(f'{OUT}/survey_pred_cell.csv').set_index('image')['count']
names = sorted(g.index)
blend = (0.5 * g.loc[names] + 0.5 * c.loc[names]).clip(lower=0)
counts = np.round(blend).astype(int)
pd.DataFrame({'image': names, 'count': counts}).to_csv(f'{OUT}/counts.csv', index=False)
print(counts.describe())
print('total', counts.sum())
