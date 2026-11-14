import pandas as pd, numpy as np, lightgbm as lgb, time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

train = pd.read_csv('challenges/task_romania/data/dataset/train.csv')
CH = [f'c{j+1:02d}' for j in range(48)]
X = train[CH].to_numpy(); y = train.vrancea.to_numpy()
skf = StratifiedKFold(5, shuffle=True, random_state=42)

oof = np.zeros(len(y))
t0 = time.time()
for fold, (tr, va) in enumerate(skf.split(X, y)):
    m = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.04, num_leaves=31, subsample=0.8,
                           colsample_bytree=0.8, min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
                           random_state=0, verbose=-1, n_jobs=4)
    m.fit(X[tr], y[tr])
    oof[va] = m.predict_proba(X[va])[:, 1]
    print(f'fold {fold} done ({time.time()-t0:.0f}s)', flush=True)

print(f'OOF auc={roc_auc_score(y,oof):.4f} acc={accuracy_score(y,(oof>0.5).astype(int)):.4f}', flush=True)

# confidence = |p-0.5|; sort descending, keep top K
order = np.argsort(-np.abs(oof - 0.5))
pred = (oof > 0.5).astype(int)
correct = (pred == y).astype(int)
err = 1 - correct
# cumulative errors among the top-K confident
cum_err = np.cumsum(err[order])
N = len(y)
# For sealed n=5000, k=4252, need 2e + s <= 748 where s = erased count.
# Scale: if we keep fraction f of 5000 (in sealed), erase 5000(1-f).
# Estimate e among kept in sealed = cum_err[K]*5000/7500 approx (same per-record rate, but need absolute counts)
# Let's compute in sealed terms: sealed has 5000 records; assume same confidence distribution.
# We have 7500 train. To simulate a "sealed-like" 5000, use the top/bottom by confidence.
def feasible_line():
    # actually simulate on train directly: find threshold point minimizing 2e+s scaled
    # Express in normalized per-5000 terms using the 7500 -> but simpler: report e vs T curve normalized
    n_keep = np.arange(1, N+1)
    m_err = cum_err
    # in sealed terms: T_sealed = n_keep * 5000/7500, e_sealed = m_err * 5000/7500, s_sealed = 5000 - T_sealed
    T = n_keep * 5000 / 7500
    e = m_err * 5000 / 7500
    s = 5000 - T
    lhs = 2*e + s
    feas = lhs <= 748
    best = np.argmin(lhs)
    print(f'min(2e+s) = {lhs[best]:.1f} at keep-top={n_keep[best]} (T_sealed={T[best]:.0f}) e={e[best]:.1f} s={s[best]:.0f}', flush=True)
    # print a few operating points
    for frac in [0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 1.0]:
        kk = int(frac * N)
        print(f'keep {frac:.2f}: T={T[kk]:.0f} e={e[kk]:.0f} s={s[kk]:.0f} 2e+s={2*e[kk]+s[kk]:.1f}', flush=True)

feasible_line()