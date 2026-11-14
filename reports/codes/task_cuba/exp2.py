"""Experiment 2: dense (provided model2vec chunk vectors), TF-IDF, fusion."""
import json, os, re, time, unicodedata
import numpy as np

DATA = "/home/kozak/mllm/MML/work/tasks/task_cuba/data/data"
BASE = "/home/kozak/mllm/MML/work/tasks/task_cuba/data/baseline/index"
OUT = "/home/kozak/mllm/MML/work/solutions/task_cuba"
STOP = set("a an the of and or to in on at by for from with as is are was were be been this that these those it its into than then there their they them he she his her we you not no but if so such which who what when where why how all any each more most other some very can could would should may might will do does did have has had about over under between through during before after".split())

def normalise(text):
    text = "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9\s]", " ", text)

def tok(text):
    return [t for t in normalise(text).split() if t not in STOP]

meta = [json.loads(l) for l in open(os.path.join(DATA, "corpus_meta.jsonl"))]
ids = [m["id"] for m in meta]
IDX = {d: i for i, d in enumerate(ids)}
docs = []
for m in meta:
    with open(os.path.join(DATA, "corpus", m["id"] + ".txt"), encoding="utf-8", errors="replace") as f:
        docs.append(f.read())
gold = {}
for l in open(os.path.join(DATA, "sample-output.jsonl")):
    r = json.loads(l); gold[r["qid"]] = r["doc_id"]
qs = [json.loads(l) for l in open(os.path.join(DATA, "sample-input.jsonl"))]
N = len(qs)

def evaluate(name, rankings):
    h1 = h3 = h10 = 0; mrr = 0.0
    for q in qs:
        g = gold[q["qid"]]; r = rankings[q["qid"]]
        h1 += g in r[:1]; h3 += g in r[:3]; h10 += g in r[:10]
        if g in r: mrr += 1.0 / (r.index(g) + 1)
    print(f"{name:<45} hit@1={h1/N:.3f} hit@3={h3/N:.3f} hit@10={h10/N:.3f} mrr={mrr/N:.3f}", flush=True)
    return h3 / N

def save(name, rankings):
    np.savez_compressed(os.path.join(OUT, "rank", name + ".npz"),
                        qids=np.array([q["qid"] for q in qs]),
                        **{q["qid"]: np.array([IDX[d] for d in rankings[q["qid"]]], dtype=np.int32) for q in qs})

results = {}
# ---------- dense ----------
t0 = time.time()
vec = np.load(os.path.join(BASE, "vectors.npy")).astype(np.float32)
cdoc = np.load(os.path.join(BASE, "chunk_doc.npy"))
from model2vec import StaticModel
model = StaticModel.from_pretrained(os.path.join(BASE, "model2vec"))
qvec = model.encode([q["question"] for q in qs]).astype(np.float32)
qvec /= np.linalg.norm(qvec, axis=1, keepdims=True) + 1e-9
S = qvec @ vec.T  # (N, n_chunks) ~ 1.8 GFLOP
print("dense scores", S.shape, f"{time.time()-t0:.1f}s", flush=True)
for agg in ("max", "sum2"):
    rank = {}
    for qi in range(N):
        ds = {}
        for c, s in enumerate(S[qi]):
            d = int(cdoc[c])
            if agg == "max":
                ds[d] = max(ds.get(d, -1e9), float(s))
            else:
                ds.setdefault(d, []).append(float(s))
        if agg == "sum2":
            ds = {d: sum(sorted(v, reverse=True)[:2]) for d, v in ds.items()}
        order = sorted(ds, key=lambda d: -ds[d])[:100]
        rank[qs[qi]["qid"]] = [ids[d] for d in order]
    results[f"dense_chunk_{agg}"] = evaluate(f"dense_chunk_{agg}", rank)
    save(f"dense_chunk_{agg}", rank)

# mean doc embedding
docvec = np.zeros((len(ids), vec.shape[1]), dtype=np.float32)
cnt = np.zeros(len(ids), dtype=np.float32)
for c, d in enumerate(cdoc):
    docvec[int(d)] += vec[c]; cnt[int(d)] += 1
docvec /= cnt[:, None] + 1e-9
docvec /= np.linalg.norm(docvec, axis=1, keepdims=True) + 1e-9
Sc = qvec @ docvec.T
rank = {qs[i]["qid"]: [ids[j] for j in np.argsort(-Sc[i])[:100]] for i in range(N)}
results["dense_docmean"] = evaluate("dense_docmean", rank)
save("dense_docmean", rank)

# ---------- tfidf word ----------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
for ngram, mindf, maxdf in (((1, 2), 2, 0.9), ((1, 3), 2, 0.9)):
    t0 = time.time()
    tv = TfidfVectorizer(lowercase=True, strip_accents="unicode", sublinear_tf=True,
                         ngram_range=ngram, min_df=mindf, max_df=maxdf, stop_words="english")
    D = tv.fit_transform(docs)
    Q = tv.transform([q["question"] for q in qs])
    D = normalize(D); Q = normalize(Q)
    Sc = (Q @ D.T).toarray()
    rank = {qs[i]["qid"]: [ids[j] for j in np.argsort(-Sc[i])[:100]] for i in range(N)}
    nm = f"tfidf_word{ngram[0]}{ngram[1]}_df{mindf}"
    results[nm] = evaluate(nm, rank)
    save(nm, rank)
    print(f"  {D.shape} {time.time()-t0:.1f}s", flush=True)

# ---------- tfidf char ----------
t0 = time.time()
tv = TfidfVectorizer(lowercase=True, strip_accents="unicode", sublinear_tf=True,
                     analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=400000,
                     max_df=0.9)
D = tv.fit_transform(docs)
Q = tv.transform([q["question"] for q in qs])
D = normalize(D); Q = normalize(Q)
Sc = (Q @ D.T).toarray()
rank = {qs[i]["qid"]: [ids[j] for j in np.argsort(-Sc[i])[:100]] for i in range(N)}
results["tfidf_char35"] = evaluate("tfidf_char35", rank)
save("tfidf_char35", rank)
print(f"  char {D.shape} {time.time()-t0:.1f}s", flush=True)

json.dump(results, open(os.path.join(OUT, "exp2_results.json"), "w"), indent=1)
