"""Zemí — retrieval over 1552 Cuba library documents.

For each question return the document ids it describes, best first.

Query-time pipeline
  1. BM25 over whole documents            -> doc score
  2. BM25 over 350-word chunks, max score -> local-evidence score
  3. int8 static embeddings over chunks   -> semantic score
  4. a small linear reranker (trained on the public dev questions) combines
     z-scored scores, score gaps, document length and source kind.

Only numpy / bm25s / model2vec are needed; the corpus is not.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata

import numpy as np

STOPWORDS = set(
    "a an the of and or to in on at by for from with as is are was were be been this that these those it its "
    "into than then there their they them he she his her we you not no but if so such which who what when where "
    "why how all any each more most other some very can could would should may might will do does did have has "
    "had about over under between through during before after".split()
)


def normalise(text: str) -> str:
    text = "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9\s]", " ", text)


def tokenize(text: str) -> list[str]:
    return [t for t in normalise(text).split() if t not in STOPWORDS]


def _zscore(a: np.ndarray) -> np.ndarray:
    return (a - a.mean()) / (a.std() + 1e-9)


class Retriever:
    def __init__(self, index_dir: str) -> None:
        import bm25s
        from model2vec import StaticModel

        with open(os.path.join(index_dir, "config.json"), encoding="utf-8") as f:
            self.cfg = json.load(f)
        with open(os.path.join(index_dir, "doc_ids.json"), encoding="utf-8") as f:
            self.doc_ids = json.load(f)
        with open(os.path.join(index_dir, "lr_model.json"), encoding="utf-8") as f:
            self.lr = json.load(f)

        self.doc_words = np.load(os.path.join(index_dir, "doc_words.npy")).astype(np.float64)
        self.doc_kind = np.load(os.path.join(index_dir, "doc_kind.npy")).astype(np.int64)
        self.chunk_doc = np.load(os.path.join(index_dir, "chunk_doc.npy"))
        self.dense_chunk_doc = np.load(os.path.join(index_dir, "dense_chunk_doc.npy")).astype(np.int64)
        vq = np.load(os.path.join(index_dir, "vectors_int8.npy"))
        scale = np.load(os.path.join(index_dir, "dense_scale.npy"))
        self.vectors = vq.astype(np.float32) * scale[None, :]
        self.model = StaticModel.from_pretrained(os.path.join(index_dir, "model2vec"))
        self.bm25_doc = bm25s.BM25.load(os.path.join(index_dir, "bm25_doc"), load_corpus=False)
        self.bm25_chunk = bm25s.BM25.load(os.path.join(index_dir, "bm25_chunk"), load_corpus=False)

        self.n_docs = len(self.doc_ids)
        self.n_kinds = len(self.cfg["kinds"])
        self.chunk_floor = float(self.cfg["chunk_floor"])
        self.lr_mean = np.array(self.lr["mean"], dtype=np.float64)
        self.lr_scale = np.array(self.lr["scale"], dtype=np.float64)
        self.lr_coef = np.array(self.lr["coef"], dtype=np.float64)
        self.lr_b0 = float(self.lr["intercept"])
        # meta features are shared by every query
        loglen = np.log1p(self.doc_words)
        lenf = self.doc_words / 1000.0
        kinds = np.zeros((self.n_docs, self.n_kinds), dtype=np.float64)
        kinds[np.arange(self.n_docs), self.doc_kind] = 1.0
        self.meta_feats = np.concatenate([loglen[:, None], lenf[:, None], kinds], axis=1)

    def _scores(self, query: str):
        toks = tokenize(query)
        nd = self.n_docs
        if not toks:
            return None
        idx, sc = self.bm25_doc.retrieve([toks], k=nd, show_progress=False)
        doc_s = np.zeros(nd, dtype=np.float64)
        doc_s[idx[0].astype(np.int64)] = np.asarray(sc[0], dtype=np.float64)

        kk = min(int(self.cfg["chunk_top"]), len(self.chunk_doc))
        cidx, csc = self.bm25_chunk.retrieve([toks], k=kk, show_progress=False)
        chunk_s = np.full(nd, self.chunk_floor, dtype=np.float64)
        cd = self.chunk_doc[cidx[0].astype(np.int64)]
        cs = np.asarray(csc[0], dtype=np.float64)
        order = np.argsort(cd, kind="stable")
        cd, cs = cd[order], cs[order]
        bounds = np.flatnonzero(np.diff(cd)) + 1
        starts = np.concatenate([[0], bounds])
        ends = np.concatenate([bounds, [len(cd)]])
        for s, e in zip(starts, ends):
            chunk_s[cd[s]] = cs[s:e].max()

        q = self.model.encode([query])[0].astype(np.float32)
        q /= np.linalg.norm(q) + 1e-9
        sims = self.vectors @ q
        dense_s = np.full(nd, -1e9, dtype=np.float64)
        np.maximum.at(dense_s, self.dense_chunk_doc, sims.astype(np.float64))
        return doc_s, chunk_s, dense_s

    def search(self, query: str, k: int = 3) -> list[str]:
        out = self._scores(query)
        if out is None:
            return self.doc_ids[:k]
        doc_s, chunk_s, dense_s = out
        zd, zc, ze = _zscore(doc_s), _zscore(chunk_s), _zscore(dense_s)
        feats = np.stack([
            zd, zc, ze,
            doc_s, chunk_s, dense_s,
            doc_s - doc_s.max(), chunk_s - chunk_s.max(), dense_s - dense_s.max(),
        ], axis=1)
        feats = np.concatenate([feats, self.meta_feats], axis=1)
        x = (feats - self.lr_mean) / self.lr_scale
        score = x @ self.lr_coef + self.lr_b0
        best = np.argsort(-score)[:k]
        return [self.doc_ids[int(i)] for i in best]
