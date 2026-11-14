"""Experiment 1: measure retrieval methods on the 268 public questions."""
import json, os, re, sys, time, unicodedata
import numpy as np

DATA = "/home/kozak/mllm/MML/work/tasks/task_cuba/data/data"
BASE = "/home/kozak/mllm/MML/work/tasks/task_cuba/data/baseline/index"
STOP = set("a an the of and or to in on at by for from with as is are was were be been this that these those it its into than then there their they them he she his her we you not no but if so such which who what when where why how all any each more most other some very can could would should may might will do does did have has had about over under between through during before after".split())

def normalise(text):
    text = "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9\s]", " ", text)

def tok(text, stop=True):
    ts = normalise(text).split()
    return [t for t in ts if not (stop and t in STOP)]

meta = [json.loads(l) for l in open(os.path.join(DATA, "corpus_meta.jsonl"))]
ids = [m["id"] for m in meta]
docs = []
for m in meta:
    with open(os.path.join(DATA, "corpus", m["id"] + ".txt"), encoding="utf-8", errors="replace") as f:
        docs.append(f.read())
print("loaded", len(docs), "docs", sum(len(d.split()) for d in docs), "words", flush=True)
gold = {}
for l in open(os.path.join(DATA, "sample-output.jsonl")):
    r = json.loads(l); gold[r["qid"]] = r["doc_id"]
qs = [json.loads(l) for l in open(os.path.join(DATA, "sample-input.jsonl"))]
print("queries", len(qs), flush=True)

def evaluate(name, rankings):
    h1 = h3 = h10 = 0; mrr = 0.0
    for q in qs:
        g = gold[q["qid"]]
        r = rankings[q["qid"]]
        if g in r[:1]: h1 += 1
        if g in r[:3]: h3 += 1
        if g in r[:10]: h10 += 1
        if g in r:
            mrr += 1.0 / (r.index(g) + 1)
    n = len(qs)
    print(f"{name:<40} hit@1={h1/n:.3f} hit@3={h3/n:.3f} hit@10={h10/n:.3f} mrr={mrr/n:.3f}", flush=True)
    return h3 / n

IDX = {d: i for i, d in enumerate(ids)}

def save(name, rankings):
    np.savez_compressed(os.path.join("/home/kozak/mllm/MML/work/solutions/task_cuba/rank", name + ".npz"),
                        qids=np.array([q["qid"] for q in qs]),
                        **{q["qid"]: np.array([IDX[d] for d in rankings[q["qid"]]], dtype=np.int32) for q in qs})

os.makedirs("/home/kozak/mllm/MML/work/solutions/task_cuba/rank", exist_ok=True)
results = {}

# ---------- BM25 doc level ----------
import bm25s
doc_toks = [tok(d) for d in docs]
title_toks = [tok(m.get("title", "")) for m in meta]
q_toks = [tok(q["question"]) for q in qs]

for use_title in (False, True):
    corpus = [t + (title_toks[i] * 3 if use_title else []) for i, t in enumerate(doc_toks)]
    for k1 in (1.2, 1.5):
        for b in (0.5, 0.75, 0.9):
            t0 = time.time()
            bm = bm25s.BM25(k1=k1, b=b, method="lucene")
            bm.index(corpus, show_progress=False)
            idx, sc = bm.retrieve(q_toks, k=100, show_progress=False)
            rank = {qs[i]["qid"]: [ids[int(j)] for j in idx[i]] for i in range(len(qs))}
            nm = f"bm25_doc_t{int(use_title)}_k1{k1}_b{b}"
            results[nm] = evaluate(nm, rank)
            if k1 == 1.5 and b == 0.75:
                save(nm, rank)
            print(f"  ({time.time()-t0:.1f}s)", flush=True)

# ---------- BM25 chunk level ----------
def make_chunks(nwords, overlap):
    chunks = []  # (doc_idx, text)
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

for nw, ov in ((120, 40), (200, 50), (350, 100)):
    t0 = time.time()
    ch = make_chunks(nw, ov)
    ch_toks = [tok(c[1]) for c in ch]
    cdoc = np.array([c[0] for c in ch], dtype=np.int32)
    bm = bm25s.BM25(k1=1.5, b=0.75, method="lucene")
    bm.index(ch_toks, show_progress=False)
    idx, sc = bm.retrieve(q_toks, k=200, show_progress=False)
    for agg in ("max", "sum2"):
        rank = {}
        for qi in range(len(qs)):
            ds = {}
            for j, s in zip(idx[qi], sc[qi]):
                d = int(cdoc[int(j)]); s = float(s)
                if agg == "max":
                    ds[d] = max(ds.get(d, -1e9), s)
                else:
                    ds.setdefault(d, []).append(s)
            if agg == "sum2":
                ds = {d: sum(sorted(v, reverse=True)[:2]) for d, v in ds.items()}
            order = sorted(ds, key=lambda d: -ds[d])[:100]
            rank[qs[qi]["qid"]] = [ids[d] for d in order]
        nm = f"bm25_chunk{nw}_{ov}_{agg}"
        results[nm] = evaluate(nm, rank)
        save(nm, rank)
    print(f"  chunk{nw} ({len(ch)} chunks) {time.time()-t0:.1f}s", flush=True)

json.dump(results, open("/home/kozak/mllm/MML/work/solutions/task_cuba/exp1_results.json", "w"), indent=1)
