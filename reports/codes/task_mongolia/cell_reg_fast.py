#!/usr/bin/env python
"""Fast final: fit cell regressor (small) and predict survey."""
import os, time
import numpy as np
import pandas as pd
import lightgbm as lgb

OUT = '/home/kozak/mllm/MML/work/solutions/task_mongolia'
t0 = time.time()
d = np.load(f'{OUT}/cell_train.npz', allow_pickle=True)
X, Y, imgids, imgcounts = d['X'], d['Y'], d['imgids'], d['imgcounts']
print('loaded', X.shape, time.time() - t0, flush=True)

m = lgb.LGBMRegressor(n_estimators=120, learning_rate=0.08, num_leaves=15,
                      min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                      n_jobs=3, verbose=-1)
m.fit(X, Y)
print('fit', time.time() - t0, flush=True)
m.booster_.save_model(f'{OUT}/cell_lgb.txt')

s = np.load(f'{OUT}/cell_survey.npz', allow_pickle=True)
Xs, simg, snames = s['X'], s['imgids'], s['names']
ps = m.predict(Xs)
counts = np.array([ps[simg == i].sum() for i in range(len(snames))])
pd.DataFrame({'image': snames, 'count': counts}).to_csv(f'{OUT}/survey_pred_cell.csv', index=False)
print('pred stats', np.percentile(counts, [0, 5, 25, 50, 75, 95, 100]), 'total', counts.sum(),
      't', time.time() - t0, flush=True)

# in-sample sanity
ptr = m.predict(X)
pc = np.array([ptr[imgids == i].sum() for i in range(len(imgcounts))])
print('train(in-sample) MAE', np.abs(pc - imgcounts).mean(), flush=True)
