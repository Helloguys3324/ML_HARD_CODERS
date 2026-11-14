import pandas as pd, numpy as np, lightgbm as lgb, time, json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

train = pd.read_csv('challenges/task_romania/data/dataset/train.csv')
CH = [f'c{j+1:02d}' for j in range(48)]
X = train[CH].to_numpy(); y = train.vrancea.to_numpy()
skf = StratifiedKFold(5, shuffle=True, random_state=42)
folds = list(skf.split(X, y))

def cv_lgb(params, tag):
    t0 = time.time()
    accs, aucs = [], []
    for tr, va in folds:
        m = lgb.LGBMClassifier(**params, n_estimators=3000, learning_rate=0.03,
                               verbose=-1, n_jobs=4, random_state=0)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric='auc',
              callbacks=[lgb.early_stopping(100, verbose=False)])
        p = m.predict_proba(X[va])[:, 1]
        accs.append(accuracy_score(y[va], (p > 0.5).astype(int)))
        aucs.append(roc_auc_score(y[va], p))
    print(f'{tag:22s} acc={np.mean(accs):.4f} auc={np.mean(aucs):.4f} std={np.std(accs):.4f} ({time.time()-t0:.0f}s)', flush=True)
    return np.mean(accs), np.mean(aucs)

res = {}
res['base'] = cv_lgb(dict(num_leaves=31, subsample=0.8, colsample_bytree=0.8, min_child_samples=20), 'LGB-base')
res['deep'] = cv_lgb(dict(num_leaves=63, subsample=0.8, colsample_bytree=0.8, min_child_samples=10), 'LGB-deep')
res['reg']  = cv_lgb(dict(num_leaves=31, subsample=0.7, colsample_bytree=0.6, min_child_samples=30, reg_alpha=1.0, reg_lambda=5.0), 'LGB-reg')
print('SUMMARY', json.dumps(res), flush=True)