"""Cloud sweep runner for task_romania.

Reads config.json (model + feature spec), trains with 5-fold stratified CV,
writes out/oof.npy, out/sealed.npy, out/metrics.json.
"""
import json, os, sys, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler, QuantileTransformer

os.makedirs('out', exist_ok=True)
cfg = json.load(open('config.json', encoding='utf-8'))
name = cfg.get('name', 'config')

# best-effort install of heavier deps
_need = cfg.get('model')
if _need in ('xgb', 'cat'):
    import subprocess, importlib
    mod = {'xgb': 'xgboost', 'cat': 'catboost'}[_need]
    try:
        importlib.import_module(mod)
    except Exception:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', mod], check=False)

train = pd.read_csv('train.csv')
sealed = pd.read_csv('sealed.csv')
CH = [f'c{j+1:02d}' for j in range(48)]
X = train[CH].to_numpy().astype(np.float64)
Xs = sealed[CH].to_numpy().astype(np.float64)
y = train['vrancea'].to_numpy()

# ---------------- features ----------------
def build_features(raw, spec):
    if spec == 'raw':
        return raw
    if spec == 'rowstats':
        m = raw.mean(1, keepdims=True); s = raw.std(1, keepdims=True)
        return np.hstack([raw, m, s, raw.min(1, keepdims=True), raw.max(1, keepdims=True)])
    if spec == 'ratios':
        b0 = raw[:, 0:16]; b1 = raw[:, 16:32]; b2 = raw[:, 32:48]
        eps = 1e-6
        r = np.hstack([b2/(b0+eps), b2/(b1+eps), b0/(b1+eps)])
        return np.hstack([raw, r])
    if spec == 'ratios_noraw':
        b0 = raw[:, 0:16]; b1 = raw[:, 16:32]; b2 = raw[:, 32:48]
        eps = 1e-6
        return np.hstack([b2/(b0+eps), b2/(b1+eps), b0/(b1+eps)])
    if spec == 'block2':
        return raw[:, 32:48]
    if spec == 'block2_ratios':
        b0 = raw[:, 0:16]; b1 = raw[:, 16:32]; b2 = raw[:, 32:48]
        eps = 1e-6
        return np.hstack([b2, b2/(b0+eps), b2/(b1+eps)])
    raise ValueError(spec)

feat_spec = cfg.get('features', 'raw')
Xf = build_features(X, feat_spec)
Xsf = build_features(Xs, feat_spec)

use_qt = cfg.get('quantile', False)
if use_qt:
    qt = QuantileTransformer(output_distribution='normal', random_state=0, n_quantiles=1000)
    Xf = qt.fit_transform(Xf)
    Xsf = qt.transform(Xsf)

# ---------------- model factory ----------------
model_name = cfg.get('model', 'lgb')

def make_model():
    p = cfg.get('params', {})
    if model_name == 'lgb':
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=p.get('n', 1500), learning_rate=p.get('lr', 0.04),
                                  num_leaves=p.get('leaves', 31), subsample=p.get('ss', 0.8),
                                  colsample_bytree=p.get('cs', 0.8), min_child_samples=p.get('mcs', 20),
                                  reg_alpha=p.get('ra', 0.1), reg_lambda=p.get('rl', 1.0),
                                  random_state=7, verbose=-1, n_jobs=2)
    if model_name == 'xgb':
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=p.get('n', 800), learning_rate=p.get('lr', 0.05),
                                 max_depth=p.get('depth', 6), subsample=p.get('ss', 0.8),
                                 colsample_bytree=p.get('cs', 0.8), tree_method='hist',
                                 random_state=7, n_jobs=2, eval_metric='logloss')
    if model_name == 'cat':
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=p.get('n', 800), learning_rate=p.get('lr', 0.05),
                                  depth=p.get('depth', 6), verbose=0, random_seed=7)
    if model_name == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=p.get('n', 500), min_samples_leaf=p.get('msl', 2),
                                      n_jobs=2, random_state=7)
    if model_name == 'et':
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(n_estimators=p.get('n', 500), min_samples_leaf=p.get('msl', 2),
                                    n_jobs=2, random_state=7)
    if model_name == 'histgb':
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=p.get('n', 300), learning_rate=p.get('lr', 0.06),
                                              max_leaf_nodes=p.get('leaves', 31),
                                              early_stopping=False, random_state=7)
    if model_name == 'svm':
        from sklearn.svm import SVC
        return SVC(C=p.get('C', 3.0), gamma=p.get('gamma', 'scale'), probability=True, random_state=7)
    if model_name == 'mlp':
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(hidden_layer_sizes=eval(p.get('hidden', '(256,256,256)')),
                             alpha=p.get('alpha', 1e-4), max_iter=p.get('max_iter', 400),
                             early_stopping=True, random_state=7)
    raise ValueError(model_name)

# ---------------- CV ----------------
skf = StratifiedKFold(5, shuffle=True, random_state=42)
t0 = time.time()
oof = np.zeros(len(y)); sp = np.zeros(len(Xs))
for tr, va in skf.split(Xf, y):
    m = make_model()
    m.fit(Xf[tr], y[tr])
    oof[va] = m.predict_proba(Xf[va])[:, 1]
    sp += m.predict_proba(Xsf)[:, 1] / 5

auc = roc_auc_score(y, oof)
acc = accuracy_score(y, (oof > 0.5).astype(int))

# confidence curve (ranked cumulative errors)
order = np.argsort(-np.abs(oof - 0.5))
pred = (oof > 0.5).astype(int)
err = (pred != y).astype(int)
cum = np.cumsum(err[order])
N = len(y)
curve = {}
for frac in [0.55, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99, 1.0]:
    kk = min(int(frac * N), N - 1)
    T = kk * 5000 / 7500; e = cum[kk] * 5000 / 7500; s = 5000 - T
    curve[str(frac)] = {'T': round(T), 'e': round(e, 1), 's': round(s, 0), '2e+s': round(2*e + s, 1)}

metrics = {'name': name, 'model': model_name, 'features': feat_spec, 'quantile': use_qt,
           'auc': float(auc), 'acc': float(acc), 'curu': curve, 'secs': round(time.time()-t0, 1)}
np.save('out/oof.npy', oof)
np.save('out/sealed.npy', sp)
json.dump(metrics, open('out/metrics.json', 'w'))
print(json.dumps({'name': name, 'model': model_name, 'features': feat_spec,
                  'auc': round(auc, 4), 'acc': round(acc, 4), 'secs': metrics['secs']}))
print('DONE')