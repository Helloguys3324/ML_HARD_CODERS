"""Experiment 3: rank fusion of saved rankings."""
import json, os, glob
import numpy as np

OUT = "/home/kozak/mllm/MML/work/solutions/task_cuba"
DATA = "/home/kozak/mllm/MML/work/tasks/task_cuba/data/data"
qs = [json.loads(l) for l in open(os.path.join(DATA, "sample-input.jsonl"))]
gold = {}
for l in open(os.path.join(DATA, "sample-output.jsonl")):
    r = json.loads(l); gold[r["qid"]] = r["doc_id"]
meta = [json.loads(l) for l in open(os.path.join(DATA, "corpus_meta.jsonl"))]
ids = [m["id"] for m in meta]
N = len(qs)
qids = [q["qid"] for q in qs]

# load all rankings
R = {}
for p in sorted(glob.glob(os.path.join(OUT, "rank", "*.npz"))):
    name = os.path.basename(p)[:-4]
    z = np.load(p)
    R[name] = {q: z[q] for q in qids}
print("available:", list(R))

def ev(name, ranking, show=True):
    h1 = h3 = h10 = 0; mrr = 0.0
    for q in qs:
        g = gold[q["qid"]]
        r = [ids[j] for j in ranking[q["qid"]]]
        h1 += g in r[:1]; h3 += g in r[:3]; h10 += g in r[:10]
        if g in r: mrr += 1.0 / (r.index(g) + 1)
    if show:
        print(f"{name:<55} hit@1={h1/N:.3f} hit@3={h3/N:.3f} hit@10={h10/N:.3f} mrr={mrr/N:.3f}", flush=True)
    return h3 / N

def rrf(parts, k=60, top=100):
    """parts: list of (name, weight)"""
    out = {}
    for q in qs:
        s = {}
        for nm, w in parts:
            for rank, d in enumerate(R[nm][q["qid"]]):
                s[int(d)] = s.get(int(d), 0.0) + w / (k + rank)
        order = sorted(s, key=lambda d: -s[d])[:top]
        out[q["qid"]] = np.array(order, dtype=np.int32)
    return out

def norm_rrf(parts, k=60, top=100):
    """normalize each method's rrf contribution by its total mass so weights are interpretable"""
    out = {}
    for q in qs:
        s = {}
        for nm, w in parts:
            rr = R[nm][q["qid"]]
            mass = sum(1.0 / (k + r) for r in range(len(rr)))
            for rank, d in enumerate(rr):
                s[int(d)] = s.get(int(d), 0.0) + w * (1.0 / (k + rank)) / mass
        order = sorted(s, key=lambda d: -s[d])[:top]
        out[q["qid"]] = np.array(order, dtype=np.int32)
    return out

# individual reference
for nm in R:
    ev(nm, R[nm])

bm = "bm25_doc_t1_k11.5_b0.75"
cand = ["bm25_doc_t0_k11.5_b0.75", "bm25_chunk200_50_sum2",
        "bm25_chunk350_100_max", "dense_chunk_max", "dense_chunk_sum2", "dense_docmean",
        "tfidf_word12_df2", "tfidf_char35"]
best = []
for nm in cand:
    ev(f"{bm} + {nm} 1:1", rrf([(bm, 1.0), (nm, 0.5)]))
    ev(f"{bm} + {nm} 1:0.25", rrf([(bm, 1.0), (nm, 0.25)]))
    ev(f"{bm} + {nm} 1:0.1", rrf([(bm, 1.0), (nm, 0.1)]))

print("\n--- multi combos ---")
combos = {
    "bm+chunk200+dense": [(bm, 1.0), ("bm25_chunk200_50_sum2", 0.4), ("dense_chunk_sum2", 0.3)],
    "bm+chunk200+dense+tfidf": [(bm, 1.0), ("bm25_chunk200_50_sum2", 0.4), ("dense_chunk_sum2", 0.3), ("tfidf_word12_df2", 0.3)],
    "bm+dense": [(bm, 1.0), ("dense_chunk_sum2", 0.4)],
    "bm+tfidf": [(bm, 1.0), ("tfidf_word12_df2", 0.4)],
    "bm+chunk+dense+char": [(bm, 1.0), ("bm25_chunk200_50_sum2", 0.3), ("dense_chunk_sum2", 0.3), ("tfidf_char35", 0.2)],
}
for nm, parts in combos.items():
    tr = rrf(parts)
    ev(nm, tr)
    np.savez_compressed(os.path.join(OUT, "rank", "fuse_" + nm + ".npz"),
                        qids=np.array(qids), **{q: tr[q] for q in qids})
