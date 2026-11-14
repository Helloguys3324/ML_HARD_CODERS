"""Build the shipped index for task_cuba (Zemí).

Outputs (submission/index/):
  bm25_doc/        bm25s index over whole documents (title repeated 3x)
  bm25_chunk/      bm25s index over 350-word / 100-word-overlap chunks
  chunk_doc.npy    int32 chunk -> document row
  doc_ids.json     document row -> corpus id
  doc_words.npy    float32 word counts
  doc_kind.npy     int8 kind code
  vectors_int8.npy int8 (n_chunks, 256) dense chunk vectors
  dense_scale.npy  float32 per-dimension scale
  model2vec/       int8 static embedding model (query encoder)
  lr_model.json    linear reranker weights + feature config
  config.json      retrieval config
"""
import json, os, re, shutil, time, unicodedata
import numpy as np

TASK = "/home/kozak/mllm/MML/work/tasks/task_cuba/data"
DATA = os.path.join(TASK, "data")
BASE = os.path.join(TASK, "baseline", "index")
SOL = "/home/kozak/mllm/MML/work/solutions/task_cuba"
OUT = os.path.join(SOL, "submission", "index")
STOP = set("a an the of and or to in on at by for from with as is are was were be been this that these those it its into than then there their they them he she his her we you not no but if so such which who what when where why how all any each more most other some very can could would should may might will do does did have has had about over under between through during before after".split())


def normalise(text: str) -> str:
    text = "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9\s]", " ", text)


def tok(text: str):
    return [t for t in normalise(text).split() if t not in STOP]


def make_chunks(docs, nwords, overlap):
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


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = [json.loads(l) for l in open(os.path.join(DATA, "corpus_meta.jsonl"))]
    ids = [m["id"] for m in meta]
    docs = []
    for m in meta:
        with open(os.path.join(DATA, "corpus", m["id"] + ".txt"), encoding="utf-8", errors="replace") as f:
            docs.append(f.read())
    words = np.array([m["words"] for m in meta], dtype=np.float32)
    kinds = sorted(set(m["kind"] for m in meta))
    kind_arr = np.array([kinds.index(m["kind"]) for m in meta], dtype=np.int8)
    json.dump(ids, open(os.path.join(OUT, "doc_ids.json"), "w"))
    np.save(os.path.join(OUT, "doc_words.npy"), words)
    np.save(os.path.join(OUT, "doc_kind.npy"), kind_arr)

    import bm25s
    t0 = time.time()
    doc_toks = [tok(d) for d in docs]
    title_toks = [tok(m.get("title", "")) for m in meta]
    corpus = [doc_toks[i] + title_toks[i] * 3 for i in range(len(docs))]
    bm = bm25s.BM25(k1=1.5, b=0.75, method="lucene")
    bm.index(corpus, show_progress=False)
    bm.save(os.path.join(OUT, "bm25_doc"), show_progress=False)
    print("doc index", round(time.time() - t0, 1), "s", flush=True)

    t0 = time.time()
    ch = make_chunks(docs, 350, 100)
    ch_toks = [tok(c[1]) for c in ch]
    cdoc = np.array([c[0] for c in ch], dtype=np.int32)
    np.save(os.path.join(OUT, "chunk_doc.npy"), cdoc)
    bmc = bm25s.BM25(k1=1.5, b=0.75, method="lucene")
    bmc.index(ch_toks, show_progress=False)
    bmc.save(os.path.join(OUT, "bm25_chunk"), show_progress=False)
    print("chunk index", len(ch), "chunks", round(time.time() - t0, 1), "s", flush=True)

    # float16 impacts halve the index size, rankings are unchanged
    for name in ("bm25_doc", "bm25_chunk"):
        p = os.path.join(OUT, name, "data.csc.index.npy")
        a = np.load(p)
        np.save(p, a.astype(np.float16))
        print(name, "impacts -> float16", a.nbytes // 1024, "->", a.astype(np.float16).nbytes // 1024, "KiB", flush=True)

    # int8 dense chunk vectors, per-dimension scale
    V = np.load(os.path.join(BASE, "vectors.npy")).astype(np.float32)
    dense_cdoc = np.load(os.path.join(BASE, "chunk_doc.npy")).astype(np.int32)
    np.save(os.path.join(OUT, "dense_chunk_doc.npy"), dense_cdoc)
    scale = (np.abs(V).max(axis=0) / 127.0).astype(np.float32)
    Vq = np.clip(np.rint(V / scale[None, :]), -127, 127).astype(np.int8)
    np.save(os.path.join(OUT, "vectors_int8.npy"), Vq)
    np.save(os.path.join(OUT, "dense_scale.npy"), scale)
    print("vectors", V.shape, "-> int8", Vq.nbytes // 1024, "KiB", flush=True)
    del V, Vq

    from model2vec import StaticModel
    from model2vec.quantization import quantize_embeddings, DType
    model = StaticModel.from_pretrained(os.path.join(BASE, "model2vec"))
    model.embedding = quantize_embeddings(model.embedding, DType.Int8)
    mdir = os.path.join(OUT, "model2vec")
    shutil.rmtree(mdir, ignore_errors=True)
    model.save_pretrained(mdir)
    print("model2vec int8 saved", sum(os.path.getsize(os.path.join(mdir, f)) for f in os.listdir(mdir)) // 1024, "KiB", flush=True)

    # floor used for documents with no chunk in the BM25 top-500 (matches training)
    chunk_floor = 1.325810194015503
    try:
        z = np.load(os.path.join(SOL, "raw_scores.npz"))
        chunk_floor = float(np.nanmin(np.where(np.isnan(z["ChMaxS"]), np.nan, z["ChMaxS"])))
    except Exception:
        pass
    json.dump({"k1": 1.5, "b": 0.75, "chunk_words": 350, "chunk_overlap": 100,
               "chunk_top": 500, "chunk_floor": chunk_floor, "kinds": kinds},
              open(os.path.join(OUT, "config.json"), "w"), indent=1)
    shutil.copy(os.path.join(SOL, "lr_final.json"), os.path.join(OUT, "lr_model.json"))
    print("chunk_floor", chunk_floor, flush=True)

    total = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(OUT) for f in fs)
    print("index total MB", round(total / 1e6, 2), flush=True)


if __name__ == "__main__":
    main()
