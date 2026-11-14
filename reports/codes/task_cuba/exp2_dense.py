"""Experiment 2: dense chunk-level retrieval with MiniLM."""
import sys, time, re
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, "/home/kozak/mllm/MML/work/solutions/task_cuba")
from lib_data import load_corpus, load_qs, evaluate

MODEL = sys.argv[1] if len(sys.argv) > 1 else "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_WORDS = int(sys.argv[2]) if len(sys.argv) > 2 else 150
STRIDE = CHUNK_WORDS // 2
PREFIX_Q = sys.argv[3] if len(sys.argv) > 3 else ""

meta, docs = load_corpus()
ids = [m["id"] for m in meta]
titles = [m.get("title") or "" for m in meta]
qs, gold = load_qs()


def chunk_text(text, cw=CHUNK_WORDS, stride=STRIDE):
    words = text.split()
    if len(words) <= cw:
        return [" ".join(words)]
    out = []
    for s in range(0, len(words), stride):
        ch = words[s:s + cw]
        if len(ch) < 40:
            break
        out.append(" ".join(ch))
        if s + cw >= len(words):
            break
    return out


chunks, chunk_doc = [], []
for di, i in enumerate(ids):
    for c in chunk_text(docs[i]):
        chunks.append(c)
        chunk_doc.append(di)
chunk_doc = np.array(chunk_doc, dtype=np.int32)
print("chunks", len(chunks), flush=True)

t0 = time.time()
model = SentenceTransformer(MODEL, cache_folder="/home/kozak/mllm/hf_cache", device="cpu")
print("model loaded", time.time() - t0, flush=True)
t0 = time.time()
V = model.encode(chunks, batch_size=128, normalize_embeddings=True, show_progress_bar=True,
                 convert_to_numpy=True)
print("encode chunks", time.time() - t0, V.shape, flush=True)
Q = model.encode([PREFIX_Q + q["question"] for q in qs], batch_size=64,
                 normalize_embeddings=True, convert_to_numpy=True)

S = Q @ V.T  # (nq, nchunks)
nq = len(qs)
ranked = {}
for qi in range(nq):
    order = np.argsort(-S[qi])[:600]
    doc_score = {}
    for cid in order:
        d = int(chunk_doc[cid])
        doc_score.setdefault(d, S[qi, cid])
    best = sorted(doc_score, key=lambda d: -doc_score[d])[:3]
    ranked[qs[qi]["qid"]] = [ids[d] for d in best]

import lib_data
print(f"{MODEL} chunk={CHUNK_WORDS} hit@3={lib_data.hit_at_k(ranked, gold):.4f}")
np.save("/tmp/opencode/dense_V.npy", V.astype(np.float16))
np.save("/tmp/opencode/dense_chunkdoc.npy", chunk_doc)
with open("/tmp/opencode/dense_chunks.txt", "w") as f:
    f.write("\n".join(chunks))
