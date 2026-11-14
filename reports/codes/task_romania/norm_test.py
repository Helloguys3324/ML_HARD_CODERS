import pandas as pd, numpy as np, lightgbm as lgb, json
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score
import time

train = pd.read_csv('challenges/task_romania/data/dataset/train.csv')
CH = [f'c{j+1:02d}' for j in range(48)]
X = train[CH].to_numpy(); y = train.vrancea.to_numpy()
skf = StratifiedKFold(5, shuffle=True, random_state=42)

def evaluate(X_use, tag):
    t0=time.time()
    p = cross_val_predict(lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, n_jobs=4, verbose=-1, random_state=0), X_use, y, cv=skf, method='predict_proba')[:,1]
    a = roc_auc_score(y,p); acc = accuracy_score(y,(p>0.5).astype(int))
    print(f'{tag:28s} auc={a:.4f} acc@0.5={acc:.4f} ({time.time()-t0:.0f}s)', flush=True)
    return a, acc

res = {}
res['raw'] = evaluate(X, 'raw')
res['L2norm'] = evaluate(X / np.linalg.norm(X, axis=1, keepdims=True), 'L2-row-norm')
res['row_z'] = evaluate((X - X.mean(1,keepdims=True))/(X.std(1,keepdims=True)+1e-6), 'row-zscore')
res['aug'] = evaluate(np.hstack([X, X.mean(1,keepdims=True), X.std(1,keepdims=True), np.linalg.norm(X,axis=1,keepdims=True)]), 'raw+rowstats')
# per-channel rank transform (RobustScaler-like)
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution='normal', random_state=0)
res['qt'] = evaluate(qt.fit_transform(X), 'quantile-normal')
print('DONE', json.dumps(res), flush=True)