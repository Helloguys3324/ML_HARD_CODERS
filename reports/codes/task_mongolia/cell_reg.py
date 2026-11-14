#!/usr/bin/env python
"""Train cell-level LGBM regressor, evaluate image counts, predict survey."""
import os
import numpy as np
import pandas as pd
import json
import lightgbm as lgb

OUT = '/home/kozak/mllm/MML/work/solutions/task_mongolia'

d = np.load(f'{OUT}/cell_train.npz', allow_pickle=True)
X, Y, imgids, imgcounts, names = d['X'], d['Y'], d['imgids'], d['imgcounts'], d['names']
n_img = len(names)

# image-level 5-fold CV
rng = np.random.default_rng(0)
perm = rng.permutation(n_img)
folds = np.zeros(n_img, int)
for k, idx in enumerate(np.array_split(perm, 5)):
    folds[idx] = k
pred_img = np.zeros(n_img)
for f in range(5):
    tr = np.isin(imgids, np.where(folds != f)[0])
    va = ~tr
    m = lgb.LGBMRegressor(n_estimators=350, learning_rate=0.05, num_leaves=31,
                          min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                          n_jobs=4, verbose=-1)
    m.fit(X[tr], Y[tr])
    pv = m.predict(X[va])
    for i in np.where(folds == f)[0]:
        pred_img[i] = pv[imgids[va] == i].sum()
    print('fold', f, 'done', flush=True)
print('CV image MAE', np.abs(pred_img - imgcounts).mean(),
      'RMSE', np.sqrt(((pred_img - imgcounts) ** 2).mean()),
      'corr', np.corrcoef(pred_img, imgcounts)[0, 1])
print('pred std', pred_img.std(), 'truth std', imgcounts.std())
np.savez(f'{OUT}/cell_cv.npz', pred=pred_img, truth=imgcounts, names=names)

# final model on all train
m = lgb.LGBMRegressor(n_estimators=350, learning_rate=0.05, num_leaves=31,
                      min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                      n_jobs=4, verbose=-1)
m.fit(X, Y)
m.booster_.save_model(f'{OUT}/cell_lgb.txt')

# predict survey if features are ready
if os.path.exists(f'{OUT}/cell_survey.npz'):
    s = np.load(f'{OUT}/cell_survey.npz', allow_pickle=True)
    Xs, simg, snames = s['X'], s['imgids'], s['names']
    ps = m.predict(Xs)
    counts = np.array([ps[simg == i].sum() for i in range(len(snames))])
    pd.DataFrame({'image': snames, 'count': counts}).to_csv(f'{OUT}/survey_pred_cell.csv', index=False)
    print('survey pred stats', np.percentile(counts, [0, 5, 25, 50, 75, 95, 100]), 'total', counts.sum())
else:
    print('survey features not ready')
