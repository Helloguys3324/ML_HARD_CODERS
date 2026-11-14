"""Experiment 1: lexical baselines at document level."""
import sys, time, re, unicodedata
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

sys.path.insert(0, "/home/kozak/mllm/MML/work/solutions/task_cuba")
from lib_data import load_corpus, load_qs, evaluate

meta, docs = load_corpus()
ids = [m["id"] for m in meta]
texts = [docs[i] for i in ids]
qs, gold = load_qs()
print("docs", len(ids), "qs", len(qs))


def run(name, vec):
    t0 = time.time()
    X = vec.fit_transform(texts)
    X = normalize(X)
    Q = normalize(vec.transform([q["question"] for q in qs]))
    S = (Q @ X.T).toarray()
    ranked = {}
    for qi, q in enumerate(qs):
        top = np.argsort(-S[qi])[:3]
        ranked[q["qid"]] = [ids[j] for j in top]
    import lib_data
    print(f"{name:45s} hit@3={lib_data.hit_at_k(ranked, gold):.4f}  ({time.time()-t0:.1f}s)")

run("tfidf word 1gram", TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True, min_df=2, stop_words="english"))
run("tfidf word 1-2gram", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, stop_words="english"))
run("tfidf word 1-2gram nostop", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2))
run("tfidf char 3-5", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=3, max_features=300000))
