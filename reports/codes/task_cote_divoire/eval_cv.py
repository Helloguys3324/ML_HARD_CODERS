"""Evaluate classifier options for task_cote_divoire via stratified CV + old-sample holdout."""
import json
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

HERE = Path(__file__).parent
DATA = HERE / "data"

ti = json.load(open(DATA / "train-input.json"))
to = json.load(open(DATA / "train-output.json"))
old = json.load(open(DATA / "old_task_sample.json"))

out = {r["id"]: r["label"] for r in to}
texts = [r["text"] for r in ti]
labels = [out[r["id"]] for r in ti]

OLD = {f"OLD_{x}" for x in ["JOIN_CUSTOMER_ORDERS", "GROUPBY_REVENUE_BY_CATEGORY",
        "CRUD_INSERT_ORDER", "CRUD_UPDATE_INVENTORY", "CRUD_DELETE_ACCOUNT"]}

def per_task_acc(y, p):
    old_c = old_t = new_c = new_t = 0
    for a, b in zip(y, p):
        if a in OLD:
            old_t += 1; old_c += (a == b)
        else:
            new_t += 1; new_c += (a == b)
    return old_c/old_t, new_c/new_t

def build(vec_kind, clf_kind):
    if vec_kind == "word":
        vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, sublinear_tf=True)
    elif vec_kind == "char":
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5), min_df=1, sublinear_tf=True)
    elif vec_kind == "wordchar":
        vec = TfidfVectorizer(ngram_range=(1,2), analyzer="word", sublinear_tf=True)
    if clf_kind == "lr":
        clf = LogisticRegression(max_iter=2000, C=5.0)
    elif clf_kind == "svc":
        clf = LinearSVC(C=1.0)
    elif clf_kind == "nb":
        clf = MultinomialNB()
    return Pipeline([("vec", vec), ("clf", clf)])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
X = np.array(texts); y = np.array(labels)

for vec in ["word", "char"]:
    for clf in ["lr", "svc", "nb"]:
        pipe = build(vec, clf)
        oa, na = [], []
        for tr, te in skf.split(X, y):
            pipe.fit(X[tr], y[tr])
            p = pipe.predict(X[te])
            o, n = per_task_acc(y[te], p)
            oa.append(o); na.append(n)
        print(f"{vec:6} {clf:4}  old={np.mean(oa):.3f}  new={np.mean(na):.3f}  "
              f"overall={np.mean([(a==b) for a,b in zip([y[i] for i in te for te_ in []],[])]) if False else ''}")

# Holdout: train on full 250, evaluate on the 29 non-overlapping old samples
print("\n--- old-sample holdout (train on 250, test on old_task_sample) ---")
for vec in ["word", "char"]:
    for clf in ["lr", "svc"]:
        pipe = build(vec, clf)
        pipe.fit(texts, labels)
        p = pipe.predict([r["text"] for r in old])
        correct = sum(1 for r, pr in zip(old, p) if r["label"] == pr)
        print(f"{vec:6} {clf:4}  old-sample acc={correct}/{len(old)} = {correct/len(old):.3f}")