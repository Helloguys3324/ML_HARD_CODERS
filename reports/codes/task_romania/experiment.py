import pandas as pd, numpy as np
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import json, time, sys

train = pd.read_csv('challenges/task_romania/data/dataset/train.csv')
CH = [f'c{j+1:02d}' for j in range(48)]
X = train[CH].to_numpy(); y = train.vrancea.to_numpy()
skf = StratifiedKFold(5, shuffle=True, random_state=42)
folds = list(skf.split(X, y))

def cv_model(name, fit_predict):
    t0 = time.time()
    accs, aucs = [], []
    for tr, va in folds:
        p = fit_predict(X[tr], y[tr], X[va])
        if p.ndim == 2: p = p[:, 1]
        accs.append(accuracy_score(y[va], (p > 0.5).astype(int)))
        aucs.append(roc_auc_score(y[va], p))
    print(f'{name:18s} acc={np.mean(accs):.4f} auc={np.mean(aucs):.4f} std={np.std(accs):.4f} ({time.time()-t0:.0f}s)', flush=True)
    return np.mean(accs), np.mean(aucs)

results = {}

def lgb_fn(Xtr, ytr, Xva):
    m = lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.03, num_leaves=31, subsample=0.8,
                           colsample_bytree=0.8, min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
                           random_state=0, verbose=-1, n_jobs=8)
    m.fit(Xtr, ytr); return m.predict_proba(Xva)

def xgb_fn(Xtr, ytr, Xva):
    m = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.03, max_depth=6, subsample=0.8,
                          colsample_bytree=0.8, tree_method='hist', random_state=0, n_jobs=8,
                          eval_metric='logloss')
    m.fit(Xtr, ytr); return m.predict_proba(Xva)

def cat_fn(Xtr, ytr, Xva):
    m = CatBoostClassifier(iterations=1200, learning_rate=0.04, depth=6, verbose=0, random_seed=0)
    m.fit(Xtr, ytr); return m.predict_proba(Xva)

for name, fn in [('LGB', lgb_fn), ('XGB', xgb_fn), ('CatBoost', cat_fn)]:
    acc, auc = cv_model(name, fn)
    results[name] = {'acc': acc, 'auc': auc}

print('SUMMARY', json.dumps(results), flush=True)