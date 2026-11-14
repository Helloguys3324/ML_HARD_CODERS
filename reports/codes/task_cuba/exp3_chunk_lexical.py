"""Experiment 3: chunk-level lexical retrieval with max-pooling."""
import sys, time, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

sys.path.insert(0, "/home/kozak/mllm/MML/work/solutions/task_cuba")
from lib_data import load_corpus, load_qs
import lib_data

meta, docs = load_corpus()
ids = [m["id"] for m in meta]
qs, gold = load_qs()
Q = [q["question"] for q in qs]


def make_chunks(cw, stride, min_words=15):
    chunks, cd = [], []
    for di, i in enumerate(ids):
        words = docs[i].split()
        if len(words) <= cw:
            if len(words) >= min_words:
                chunks.append(" ".join(words)); cd.append(di)
            continue
        for s in range(0, len(words), stride):
            ch = words[s:s + cw]
            if len(ch) < min_words:
                break
            chunks.append(" ".join(ch)); cd.append(di)
            if s + cw >= len(words):
                break
    return chunks, np.array(cd, dtype=np.int32)


def evaluate_chunks(cw, stride, vec_kw, topc=300, w_doc=1.0, label=""):
    t0 = time.time()
    chunks, cd = make_chunks(cw, stride)
    vec = TfidfVectorizer(**vec_kw)
    X = normalize(vec.fit_transform(chunks))
    Qm = normalize(vec.transform(Q))
    S = (Qm @ X.T).toarray()          # (nq, nchunks)
    nchunks = len(chunks)
    ranked = {}
    for qi in range(len(qs)):
        s = S[qi]
        order = np.argpartition(-s, min(topc, nchunks - 1))[:topc]
        ds = {}
        for cid in order:
            d = int(cd[cid])
            if s[cid] > ds.get(d, -1):
                ds[d] = s[cid]
        best = sorted(ds, key=lambda d: -ds[d])[:3]
        ranked[qs[qi]["qid"]] = [ids[d] for d in best]
    h = lib_data.hit_at_k(ranked, gold)
    print(f"{label:30s} cw={cw} stride={stride} chunks={nchunks:6d} hit@3={h:.4f} time={time.time()-t0:.1f}s", flush=True)
    return h, ranked


vec_base = dict(ngram_range=(1, 2), sublinear_tf=True, min_df=2, stop_words="english")
for cw in (60, 120, 250, 500):
    evaluate_chunks(cw, cw // 2, vec_base, label="tfidf 1-2gram")

vec_nostop = dict(ngram_range=(1, 2), sublinear_tf=True, min_df=2, stop_words=None)
evaluate_chunks(120, 60, vec_nostop, label="tfidf 1-2gram nostop")
