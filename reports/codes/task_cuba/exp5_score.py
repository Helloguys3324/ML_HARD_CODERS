"""Experiment 5: score-level fusion + priors."""
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
ND = len(ids)
IDX = {d: i for i, d in enumerate(ids)}
docs = []
for m in meta:
    with open(os.path.join(DATA, "corpus", m["id"] + ".txt"), encoding="utf-8", errors="replace") as f:
        docs.append(f.read())
words = np.array([m["words"] for m in meta], dtype=np.float32)
kinds = sorted(set(m["kind"] for m in meta))
KIND = {k: i for i, k in enumerate(kinds)}
kind_arr = np.array([KIND[m["kind"]] for m in meta])
gold = {}
for l in open(os.path.join(DATA, "sample-output.jsonl")):
    r = json.loads(l); gold[r["qid"]] = r["doc_id"]
qs = [json.loads(l) for l in open(os.path.join(DATA, "sample-input.jsonl"))]
N = len(qs)
qids = [q["qid"] for q in qs]

def evaluate(name, ranking, show=True):
    h1 = h3 = h10 = 0; mrr = 0.0
    for q in qs:
        g = gold[q["qid"]]; r = ranking[q["qid"]]
        h1 += g in r[:1]; h3 += g in r[:3]; h10 += g in r[:10]
        if g in r: mrr += 1.0 / (r.index(g) + 1)
    if show:
        print(f"{name:<55} hit@1={h1/N:.3f} hit@3={h3/N:.3f} hit@10={h10/N:.3f} mrr={mrr/N:.3f}", flush=True)
    return h3 / N

def save(name, ranking):
    np.savez_compressed(os.path.join(OUT, "rank", name + ".npz"),
                        qids=np.array(qids),
                        **{q: np.array([IDX[d] for d in ranking[q]], dtype=np.int32) for q in qids})

import bm25s
t0 = time.time()
doc_toks = [tok(d) for d in docs]
title_toks = [tok(m.get("title", "")) for m in meta]
corpus = [doc_toks[i] + title_toks[i] * 3 for i in range(ND)]
qt = [tok(q["question"]) for q in qs]
bm = bm25s.BM25(k1=1.5, b=0.75, method="lucene")
bm.index(corpus, show_progress=False)
idx, sc = bm.retrieve(qt, k=ND, show_progress=False)
DocS = np.full((N, ND), -1e9, dtype=np.float32)
for qi in range(N):
    DocS[qi, idx[qi]] = sc[qi]
print("doc bm25", f"{time.time()-t0:.1f}s", flush=True)

# chunk-level 350/100
def make_chunks(nwords, overlap):
    chunks = []
    for di, d in enumerate(docs):
        w = d.split()
        if len(w) <= nwords:
            chunks.append((di, " ".join(w)))
        else:
            step = nwords - overlap
            for s in range(0, len(w), step):
                seg = w[s:s + nwords]
                if len(seg) < 30 and s > 0:
                    break
                chunks.append((di, " ".join(seg)))
    return chunks

t0 = time.time()
ch = make_chunks(350, 100)
ch_toks = [tok(c[1]) for c in ch]
cdoc = np.array([c[0] for c in ch], dtype=np.int32)
bmc = bm25s.BM25(k1=1.5, b=0.75, method="lucene")
bmc.index(ch_toks, show_progress=False)
cidx, csc = bmc.retrieve(qt, k=500, show_progress=False)
ChMaxS = np.full((N, ND), np.nan, dtype=np.float32)
for qi in range(N):
    for j, s in zip(cidx[qi], csc[qi]):
        d = int(cdoc[int(j)])
        if np.isnan(ChMaxS[qi, d]) or s > ChMaxS[qi, d]:
            ChMaxS[qi, d] = s
print("chunk bm25", f"{time.time()-t0:.1f}s", flush=True)

# dense
t0 = time.time()
vec = np.load(os.path.join(BASE, "vectors.npy")).astype(np.float32)
bcdoc = np.load(os.path.join(BASE, "chunk_doc.npy"))
from model2vec import StaticModel
model = StaticModel.from_pretrained(os.path.join(BASE, "model2vec"))
qvec = model.encode([q["question"] for q in qs]).astype(np.float32)
qvec /= np.linalg.norm(qvec, axis=1, keepdims=True) + 1e-9
S = qvec @ vec.T
DenseS = np.full((N, ND), -1e9, dtype=np.float32)
for qi in range(N):
    for c in range(S.shape[1]):
        d = int(bcdoc[c])
        if S[qi, c] > DenseS[qi, d]:
            DenseS[qi, d] = S[qi, c]
print("dense", f"{time.time()-t0:.1f}s", flush=True)

np.savez_compressed(os.path.join(OUT, "raw_scores.npz"), DocS=DocS, ChMaxS=ChMaxS, DenseS=DenseS)

def zs(A):
    m = np.nanmean(A, axis=1, keepdims=True)
    s = np.nanstd(A, axis=1, keepdims=True) + 1e-9
    return (A - m) / s

ZD = zs(DocS)
ZC = zs(np.nan_to_num(ChMaxS, nan=np.nanmin(ChMaxS)))
ZE = zs(DenseS)
loglen = (np.log(words / np.mean(words))).astype(np.float32)

def rank_from(scores, extra=None):
    Sx = scores if extra is None else scores + extra[None, :]
    order = np.argsort(-Sx, axis=1)[:, :100]
    return {qids[i]: [ids[j] for j in order[i]] for i in range(N)}

best = (0, None)
for wc in (0.0, 0.2, 0.35, 0.5, 0.7, 1.0):
    for we in (0.0, 0.1, 0.2, 0.3):
        r = rank_from(ZD + wc * ZC + we * ZE)
        h3 = evaluate(f"zscore wc={wc} we={we}", r, show=False)
        if h3 > best[0]:
            best = (h3, (wc, we))
        print(f"  wc={wc} we={we} -> {h3:.3f}", flush=True)
print("best", best)
wc, we = best[1]
base = ZD + wc * ZC + we * ZE
print("\n--- length prior ---")
for al in (-0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5):
    r = rank_from(base, al * loglen)
    evaluate(f"len alpha={al}", r)
print("\n--- kind prior ---")
for kb in (0.0, 0.05, 0.1, 0.2):
    extra = np.zeros(ND, dtype=np.float32)
    # dev-fitted: book chapters and report sections up-weighted, wiki down
    for k in kinds:
        pass
    # compute optimal simple bias later; just test book+report boost
    extra[kind_arr != KIND["wiki_article"]] = kb
    r = rank_from(base, extra)
    evaluate(f"nonwiki boost={kb}", r)
