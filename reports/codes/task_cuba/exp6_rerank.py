"""Experiment 6: supervised reranker over first-stage candidates (CV by query)."""
import json, os, glob
import numpy as np

OUT = "/home/kozak/mllm/MML/work/solutions/task_cuba"
DATA = "/home/kozak/mllm/MML/work/tasks/task_cuba/data/data"
meta = [json.loads(l) for l in open(os.path.join(DATA, "corpus_meta.jsonl"))]
ids = [m["id"] for m in meta]
ND = len(ids)
IDX = {d: i for i, d in enumerate(ids)}
words = np.array([m["words"] for m in meta], dtype=np.float32)
kinds = sorted(set(m["kind"] for m in meta))
KIND = {k: i for i, k in enumerate(kinds)}
kind_arr = np.array([KIND[m["kind"]] for m in meta])
qs = [json.loads(l) for l in open(os.path.join(DATA, "sample-input.jsonl"))]
gold = {}
for l in open(os.path.join(DATA, "sample-output.jsonl")):
    r = json.loads(l); gold[r["qid"]] = r["doc_id"]
N = len(qs)
qids = [q["qid"] for q in qs]

Z = np.load(os.path.join(OUT, "raw_scores.npz"))
DocS, ChMaxS, DenseS = Z["DocS"], np.nan_to_num(Z["ChMaxS"], nan=np.nanmin(Z["ChMaxS"])), Z["DenseS"]

def zs(A):
    return (A - A.mean(axis=1, keepdims=True)) / (A.std(axis=1, keepdims=True) + 1e-9)
ZD, ZC, ZE = zs(DocS), zs(ChMaxS), zs(DenseS)

# rankings from all methods
R = {}
for p in sorted(glob.glob(os.path.join(OUT, "rank", "*.npz"))):
    if "fuse_" in p: continue
    nm = os.path.basename(p)[:-4]
    z = np.load(p)
    arr = np.full((N, ND), 1000, dtype=np.int32)
    for qi, q in enumerate(qs):
        for rank, d in enumerate(z[q["qid"]]):
            arr[qi, int(d)] = rank
    R[nm] = arr

methods = list(R)
cand_methods = ["bm25_doc_t1_k11.5_b0.75", "bm25_chunk350_100_max", "bm25_chunk200_50_sum2",
                "dense_chunk_max", "tfidf_word12_df2", "bm25_chunk500_150_max", "bm25_doc_stem_k12.0_b0.85"]

# candidate union top 60
cand_mask = np.zeros((N, ND), dtype=bool)
for nm in cand_methods:
    cand_mask |= (R[nm] < 60)
print("avg candidates per query", cand_mask.sum(axis=1).mean())

# feature matrix in long format
rows_q, rows_d, feats = [], [], []
feat_names = None
for qi, q in enumerate(qs):
    ds = np.where(cand_mask[qi])[0]
    if len(ds) == 0: continue
    F = []
    for d in ds:
        f = []
        f += [ZD[qi, d], ZC[qi, d], ZE[qi, d]]
        f += [DocS[qi, d], ChMaxS[qi, d], DenseS[qi, d]]
        f += [DocS[qi, d] - DocS[qi].max(), ChMaxS[qi, d] - ChMaxS[qi].max(), DenseS[qi, d] - DenseS[qi].max()]
        for nm in methods:
            r = R[nm][qi, d]
            f += [min(r, 200) / 200.0, 1.0 / (1 + r)]
        f += [np.log1p(words[d]), words[d] / 1000.0]
        f += [float(kind_arr[d] == k) for k in range(len(kinds))]
        F.append(f)
    F = np.array(F, dtype=np.float32)
    rows_q.append(np.full(len(ds), qi, dtype=np.int32))
    rows_d.append(ds)
    feats.append(F)
    if feat_names is None:
        feat_names = (["zD", "zC", "zE", "rawD", "rawC", "rawE", "gapD", "gapC", "gapE"]
                      + [f"{nm}_{s}" for nm in methods for s in ("r", "rr")]
                      + ["loglen", "len"] + ["kind_" + k for k in kinds])

X = np.vstack(feats)
qi_arr = np.concatenate(rows_q)
doc_arr = np.concatenate(rows_d)
y = np.array([1 if ids[d] == gold[qids[q]] else 0 for q, d in zip(qi_arr, doc_arr)], dtype=np.int8)
print("X", X.shape, "pos", y.sum(), "features", len(feat_names))

# baseline z-fusion hit3
base = ZD + 0.2 * ZC + 0.3 * ZE
def hit3_from_scores(score_fn, name):
    h3 = h1 = 0
    for qi in range(N):
        ds = np.where(cand_mask[qi])[0]
        s = score_fn(qi, ds)
        top = ds[np.argsort(-s)[:3]]
        g = gold[qids[qi]]
        h3 += g in [ids[d] for d in top]
        h1 += ids[top[0]] == g
    print(f"{name:<30} hit@1={h1/N:.3f} hit@3={h3/N:.3f}", flush=True)
    return h3 / N

hit3_from_scores(lambda qi, ds: base[qi, ds], "z-fusion baseline")

import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

gkf = GroupKFold(n_splits=5)
oof_lgb = np.zeros(len(y), dtype=np.float32)
oof_lr = np.zeros(len(y), dtype=np.float32)
for fold, (tr, va) in enumerate(gkf.split(X, y, groups=qi_arr)):
    import time as _t
    _t0 = _t.time()
    m = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=15,
                           min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                           random_state=0, verbosity=-1, n_jobs=1)
    m.fit(X[tr], y[tr])
    oof_lgb[va] = m.predict_proba(X[va])[:, 1]
    _t1 = _t.time()
    sc = StandardScaler().fit(X[tr])
    lr = LogisticRegression(C=0.1, max_iter=1000)
    lr.fit(sc.transform(X[tr]), y[tr])
    oof_lr[va] = lr.predict_proba(sc.transform(X[va]))[:, 1]
    print("fold", fold, "done", f"lgb={_t1-_t0:.1f}s lr={_t.time()-_t1:.1f}s", flush=True)

# aggregate OOF scores into per-query arrays
SL = np.full((N, ND), -1e9, dtype=np.float32)
SR = np.full((N, ND), -1e9, dtype=np.float32)
for k in range(len(y)):
    SL[qi_arr[k], doc_arr[k]] = oof_lgb[k]
    SR[qi_arr[k], doc_arr[k]] = oof_lr[k]
hit3_from_scores(lambda qi, ds: SL[qi, ds], "LGBM OOF")
hit3_from_scores(lambda qi, ds: SR[qi, ds], "LR OOF")
hit3_from_scores(lambda qi, ds: base[qi, ds] + 0.5 * SR[qi, ds], "z-fusion + 0.5 LR")
hit3_from_scores(lambda qi, ds: base[qi, ds] + 1.0 * SR[qi, ds], "z-fusion + 1.0 LR")

# feature importance full model
m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=15,
                       min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                       random_state=0, verbosity=-1, n_jobs=4)
m.fit(X, y)
imp = sorted(zip(feat_names, m.feature_importances_), key=lambda x: -x[1])
print("\nTop features:")
for nm, v in imp[:25]:
    print(f"  {nm:<30} {v}")
