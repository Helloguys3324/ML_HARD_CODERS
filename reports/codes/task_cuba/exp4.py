"""Experiment 4: stemming, bigrams, larger chunks, BM25 param tuning."""
import json, os, re, time, unicodedata
import numpy as np

DATA = "/home/kozak/mllm/MML/work/tasks/task_cuba/data/data"
OUT = "/home/kozak/mllm/MML/work/solutions/task_cuba"
STOP = set("a an the of and or to in on at by for from with as is are was were be been this that these those it its into than then there their they them he she his her we you not no but if so such which who what when where why how all any each more most other some very can could would should may might will do does did have has had about over under between through during before after".split())

def normalise(text):
    text = "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9\s]", " ", text)

import Stemmer
stemmer = Stemmer.Stemmer("english")

def tok(text, stem=False, bigrams=0):
    ts = [t for t in normalise(text).split() if t not in STOP]
    if stem:
        ts = stemmer.stemWords(ts)
    if bigrams:
        bg = ["bg_" + ts[i] + "_" + ts[i + 1] for i in range(len(ts) - 1)]
        ts = ts + bg * bigrams
    return ts

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

def evaluate(name, rankings, keep=False):
    h1 = h3 = h10 = 0; mrr = 0.0
    for q in qs:
        g = gold[q["qid"]]; r = rankings[q["qid"]]
        h1 += g in r[:1]; h3 += g in r[:3]; h10 += g in r[:10]
        if g in r: mrr += 1.0 / (r.index(g) + 1)
    print(f"{name:<50} hit@1={h1/N:.3f} hit@3={h3/N:.3f} hit@10={h10/N:.3f} mrr={mrr/N:.3f}", flush=True)
    return h3 / N

def save(name, rankings):
    np.savez_compressed(os.path.join(OUT, "rank", name + ".npz"),
                        qids=np.array([q["qid"] for q in qs]),
                        **{q["qid"]: np.array([IDX[d] for d in rankings[q["qid"]]], dtype=np.int32) for q in qs})

import bm25s
results = {}

def run_bm25(name, corpus_tokens, q_tokens, k1, b, k=100):
    t0 = time.time()
    bm = bm25s.BM25(k1=k1, b=b, method="lucene")
    bm.index(corpus_tokens, show_progress=False)
    idx, sc = bm.retrieve(q_tokens, k=k, show_progress=False)
    rank = {qs[i]["qid"]: [ids[int(j)] for j in idx[i]] for i in range(N)}
    results[name] = evaluate(name, rank)
    save(name, rank)
    print(f"  ({time.time()-t0:.1f}s)", flush=True)
    return rank

title_toks = [tok(m.get("title", "")) for m in meta]

# stemming variants
for stem in (True,):
    dt = [tok(d, stem=stem) for d in docs]
    qt = [tok(q["question"], stem=stem) for q in qs]
    tt = [tok(m.get("title", ""), stem=stem) for m in meta]
    for k1 in (1.2, 1.5, 2.0):
        for b in (0.6, 0.75, 0.85):
            corpus = [dt[i] + tt[i] * 3 for i in range(len(docs))]
            run_bm25(f"bm25_doc_stem_k1{k1}_b{b}", corpus, qt, k1, b)

# bigram variants (no stemming)
for nbg in (1, 2):
    dt = [tok(d, bigrams=nbg) for d in docs]
    qt = [tok(q["question"], bigrams=nbg) for q in qs]
    tt = [tok(m.get("title", ""), bigrams=nbg) for m in meta]
    corpus = [dt[i] + tt[i] * 3 for i in range(len(docs))]
    run_bm25(f"bm25_doc_bg{nbg}", corpus, qt, 1.5, 0.75)

# larger chunks
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

for nw, ov in ((500, 150), (800, 200)):
    t0 = time.time()
    ch = make_chunks(nw, ov)
    ch_toks = [tok(c[1]) for c in ch]
    cdoc = np.array([c[0] for c in ch], dtype=np.int32)
    bm = bm25s.BM25(k1=1.5, b=0.75, method="lucene")
    bm.index(ch_toks, show_progress=False)
    idx, sc = bm.retrieve([tok(q["question"]) for q in qs], k=200, show_progress=False)
    for agg in ("max", "sum2"):
        rank = {}
        for qi in range(N):
            ds = {}
            for j, s in zip(idx[qi], sc[qi]):
                d = int(cdoc[int(j)]); s = float(s)
                if agg == "max": ds[d] = max(ds.get(d, -1e9), s)
                else: ds.setdefault(d, []).append(s)
            if agg == "sum2": ds = {d: sum(sorted(v, reverse=True)[:2]) for d, v in ds.items()}
            order = sorted(ds, key=lambda d: -ds[d])[:100]
            rank[qs[qi]["qid"]] = [ids[d] for d in order]
        nm = f"bm25_chunk{nw}_{ov}_{agg}"
        results[nm] = evaluate(nm, rank)
        save(nm, rank)
    print(f"  chunk{nw} ({len(ch)} chunks) {time.time()-t0:.1f}s", flush=True)

json.dump(results, open(os.path.join(OUT, "exp4_results.json"), "w"), indent=1)
