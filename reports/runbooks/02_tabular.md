# Runbook 02 — Tabular (Europe)

> Template: `templates/02_tabular_fast.py` · Track key: `tabular`
> CPU verdict: **best track for this machine.** LightGBM/CatBoost/Polars are the
> right tools; no GPU needed. Spend time on features and validation, not tuning.
> Modes: `tabular` (default) and `unsupervised` (clustering/PCA/outliers).
> `--use-gpu` defaults on; `--no-use-gpu` is valid and avoids CUDA probing.

---

## 1. Fastest credible baseline (2–5 min)

```bash
export PY=/home/kozak/mllm/py REPO=/home/kozak/mllm/MML
cd "$REPO"
$PY lmml_master.py new lvl_05 --track tabular
$PY lmml_master.py solve lvl_05 -- \
  --train data/train.csv --test data/test.csv --sample-sub data/sample_submission.csv \
  --target target --id-col id --task auto --metric auto --cv stratified \
  --models lgb --folds 3 --max-trees 400 --no-use-gpu
$PY lmml_master.py check lvl_05 --sample lvl_05/data/sample_submission.csv
```

This fits one LightGBM per fold with the template's automatic preprocessing
(categorical encoding, frequency/cross features, OOF target encoding). On 50k–200k
rows it finishes in 2–5 min. Submit immediately.

Full run (only after the baseline is safe):

```bash
$PY lmml_master.py solve lvl_05 -- \
  --train data/train.csv --test data/test.csv --sample-sub data/sample_submission.csv \
  --target target --id-col id --models lgb,xgb,cat,ridge \
  --folds 5 --max-trees 1200 --calibrate auto --tune-threshold --no-use-gpu
```

---

## 2. Best CPU model

| Model | Role | Notes |
| :--- | :--- | :--- |
| **LightGBM** | primary | fastest on CPU; `num_leaves` 31–127, `learning_rate` 0.03–0.05, early stopping |
| **CatBoost** | categorical-heavy data | strong OOF TE; `thread_count=4`, `allow_writing_files=False` |
| **XGBoost hist** | diversity | `tree_method="hist"`, `device="cpu"`, `max_depth` 5–8 |
| **Ridge / ElasticNet** | linear diversity | one-hot or TE features; cheap, often +0.002 in blend |
| **MLP (2–3 layers)** | non-linear diversity | only if rows > 20k and time > 60 min; rank-gauss + early stop |

A 3-family blend (LGB + Cat + Ridge) is usually better than 5 seeds of LGB. Use
`optimizations/blender.auto_blend`.

---

## 3. Feature engineering (highest ROI first)

1. **Group-relative features**: for each entity group (`user_id`, `store_id`,
   `device`, `date`): `x - mean`, `x / mean`, `(x-mean)/std`, percentile rank,
   `nunique`, group size, time since previous event. This is the single biggest
   tabular win.
2. **Frequency encoding** for high-cardinality categoricals, plus
   `freq_train / (freq_test + 1)` to expose drift.
3. **Leak-free OOF target encoding** with M-estimate smoothing (`m` 5–50), noise
   only on train OOF. Use `fast_oof_target_encode` or template 02. For time series,
   use causal expanding-window TE, never K-fold TE.
4. **Row meta-features**: row mean/std/min/max, count of nulls, count of zeros,
   count of negatives, sum of absolute values, top-1 value frequency. Cheap and
   surprisingly strong.
5. **Datetime expansion**: year/month/day/dayofweek/hour, weekend flag, days since
   min, cyclical sin/cos, holiday proximity if a calendar is provided.
6. **Interactions**: only after the above; 2-way crosses of the top-8 categoricals
   with support ≥ 10, then OOF TE. Do not explode hundreds of crosses.
7. **Adversarial pruning**: drop features with adversarial AUC contribution and no
   CV gain. Template runs adversarial validation.

---

## 4. Validation

- Match the test split: `StratifiedKFold` for i.i.d. classification,
  `KFold` for regression, `GroupKFold` for entities, `TimeSeriesSplit` for time.
- Never shuffle time-ordered data. For panel data, split by time **and** group.
- Check train/test ID overlap and duplicate rows (`fast_eda.py` does this). If
  duplicate IDs exist across train/test, the task may be a lookup, not ML.
- Compute adversarial AUC on train vs test. AUC > 0.7 means shift; inspect top
  features, consider dropping or reweighting.
- Use nested OOF for any post-processing (threshold, calibration, blend weights).
  Fitting a threshold on the full OOF and reporting that score is leakage.

---

## 5. Common pitfalls

| Pitfall | Symptom | Fix |
| :--- | :--- | :--- |
| Target encoding leakage | CV 0.95, LB 0.75 | OOF TE inside folds; smoothing; noise on train only |
| ID/time leakage | One feature dominates importance | Drop IDs, use time only for split |
| Wrong metric direction | Great CV, terrible LB | Read metric; RMSE/MAE minimize, AUC/F1 maximize |
| Constant predictions | `check` fails | Ensure model trained; fallback prior with `--allow-constant` |
| CatBoost writes `catboost_info/` | Repo pollution | `allow_writing_files=False` (template default) |
| Too many Optuna trials | 40 min for +0.0005 | 10–20 trials, fold-0 pruning, fixed budget |
| Float64 memory blowup | OOM on 1M×200 | Polars `Float32`/`Int32`, drop unused columns |
| Threshold tuned on full OOF | LB drop | Nested OOF threshold |

---

## 6. Snippets

### 6.1 Minimal LightGBM baseline

```python
import numpy as np, polars as pl, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

tr = pl.read_csv("data/train.csv"); te = pl.read_csv("data/test.csv")
y = tr["target"].to_numpy(); ids = te["id"].to_numpy()
drop_cols = [c for c in ("target", "id") if c in tr.columns]
X = tr.drop(drop_cols).to_pandas(); Xte = te.drop([c for c in ("id",) if c in te.columns]).to_pandas()
# Basic categorical handling: cast strings to category codes
for c in X.columns:
    if X[c].dtype == object or str(X[c].dtype).startswith("str"):
        cats = {v: i for i, v in enumerate(sorted(set(X[c]) | set(Xte[c])))}
        X[c] = X[c].map(cats).astype("category"); Xte[c] = Xte[c].map(cats).astype("category")
skf = StratifiedKFold(5, shuffle=True, random_state=42)
oof = np.zeros(len(y)); pred = np.zeros(len(te))
for i_tr, i_va in skf.split(X, y):
    m = lgb.LGBMClassifier(n_estimators=1200, learning_rate=0.04, num_leaves=63,
                           subsample=0.8, colsample_bytree=0.8, n_jobs=4,
                           force_col_wise=True, verbose=-1)
    m.fit(X.iloc[i_tr], y[i_tr], eval_set=[(X.iloc[i_va], y[i_va])],
          callbacks=[lgb.early_stopping(80, verbose=False)])
    oof[i_va] = m.predict_proba(X.iloc[i_va])[:, 1]
    pred += m.predict_proba(Xte)[:, 1] / 5
print("OOF AUC", roc_auc_score(y, oof))
pl.DataFrame({"id": ids, "target": pred}).write_csv("submissions/submission.csv")
```

### 6.2 Rust OOF target encoding

```python
from optimizations.rust_fast_ext import fast_oof_target_encode
cats = tr["high_card_col"].cast(pl.Categorical).to_physical().to_numpy()
folds = np.zeros(len(tr), dtype=np.int32)
for f, (_, va) in enumerate(skf.split(np.zeros(len(y)), y)): folds[va] = f
te_col = fast_oof_target_encode(cats, y, folds, smooth=10.0)   # 50k rows ~15 ms
```

### 6.3 Blend and calibrate

```python
from optimizations.blender import auto_blend
res = auto_blend([oof_lgb, oof_cat, oof_ridge], y, [pred_lgb, pred_cat, pred_ridge],
                 metric="auc")
print(res["best_method"], res["best_score"], res["leaderboard"])
# For logloss use metric="logloss"; for regression task="regression", metric="rmse".
```

---

## 7. Runtime on 4 cores (shared box)

| Operation | Time |
| :--- | :--- |
| LGB 50k×40, 100 rounds | 0.5–2 s |
| LGB 200k×60, 300 rounds | 30–90 s |
| LGB 1M×30, 300 rounds | 2–5 min |
| 5-fold LGB + XGB + Cat, 100k×100 | 10–25 min |
| + Ridge/MLP + blend | +10–20 min |
| 20 Optuna trials, 3-fold LGB | 15–30 min |
| Full template demo (all 3 tasks) | ~81 s on this box |

## 8. Do not attempt without GPU

- 10-fold × 5-seed × 5-model grids (hundreds of fits).
- Deep tabular nets (FT-Transformer, TabNet, NODE) on >200k rows.
- GPU-only XGBoost `device="cuda"` or CatBoost `task_type="GPU"`.
- Optuna >30 trials or expensive hyperparameter search before features are done.
