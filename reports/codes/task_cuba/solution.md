# task_cuba — "If you know, you know!" (Hard, 300, file submission)

## Result

A complete retrieval submission is packaged at:

```
/home/kozak/mllm/MML/work/solutions/task_cuba/task_cuba.zip   (25.5 MB)
```

The archive contains exactly `main.py` + `index/` at the top level (no `.py`/`.so`/`.pyc`/symlinks in `index/`). `answer.txt` points to it; there is no flag to type — the grader prints `SIGMOID_<WORD>` only when all four limits pass on the hidden questions.

`local_check.py` on the public 268 questions (the archive, not just the folder):

| measure | ours | limit | result |
|---|---|---|---|
| index size (main.py + index/) | **41.18 MB** | ≤ 53.0 MB | PASS |
| load time `Retriever(index)` | **1.9–2.3 s** | ≤ 10.0 s | PASS |
| p95 `search()` latency | **22–37 ms** | ≤ 75.0 ms | PASS |
| Hit@3 | **0.858** | ≥ 0.80 | PASS |

Hit@1 = 0.694, Hit@10 = 0.937. Per source kind: book chapters 116/147 = 0.789, Wikipedia 101/108 = 0.935, report sections 9/9, OCR chapters 4/4. Five-fold GroupKFold (by question) on the same dev set gives **0.866 Hit@3**, i.e. the deployed reranker is not memorising the dev questions.

Baseline for comparison (supplied `baseline/`): 80.25 MB, 0.709 Hit@3, FAIL on size and accuracy.

## Approach

The corpus has 1552 long, heterogeneous documents (470 book chapters, 955 Wikipedia articles, 105 report sections, 14 noisy OCR chapters, 8 tables) and questions are long, detail-dense paraphrases ("the part where…", "what did X say about Y and Z"). Pure dense retrieval is weak on this corpus (provided model2vec: 0.58 Hit@3), pure document BM25 is strong (0.806). The final system combines three complementary signals and a small supervised reranker.

Index build (`build_index.py`, run once offline; the corpus is not needed at query time):

1. **Document BM25** (`bm25s`, Lucene method, k1=1.5, b=0.75) over the whole document, with the metadata title appended three times. 0.806 Hit@3 alone.
2. **Chunk BM25** over 350-word windows with 100-word overlap (16,246 chunks); per document we keep the best-scoring chunk. Locates the passage the slip describes rather than the whole chapter. 0.787 Hit@3 alone, but complements document BM25 (fusion 0.832).
3. **Dense chunk retrieval** with the static `model2vec` model shipped in `baseline/index/` (potion-base-8M, 256-d, 26,354 × 100-word/50-overlap chunk vectors); per document max cosine. 0.578 Hit@3 alone; adds semantic recall.

A 16-feature **logistic reranker** (C=0.03, trained on the 268 public dev pairs) scores every document with: z-scores of the three signals, raw scores, gaps to the per-query maximum, log/raw document length, and a five-way source-kind one-hot. Per-query z-scoring makes the weights independent of query length. Five-fold by-question CV: 0.866 Hit@3 (0.851 for the best hand-tuned weighted fusion without the reranker). It is deployed as plain JSON coefficients and evaluated with numpy.

### Size / speed engineering

Everything is quantised in formats the runtime libraries load natively:

- BM25 impact arrays converted from float32 to **float16** (rankings unchanged on validation queries; halves 14.2 MB → 7.1 MB).
- Dense chunk vectors: **int8 with a per-dimension scale** (26.4 MB → 6.6 MB); dense CV Hit@3 unchanged.
- `model2vec` embedding matrix: **int8** via its own quantiser (29 MB → 7.6 MB, query cosine vs float model 0.9996). The int8 model is saved with `model2vec`'s native format, so `StaticModel.from_pretrained` loads it directly.
- No corpus, no network, no PyTorch. `main.py` imports only `numpy`, `bm25s`, `model2vec` (all in the grading runtime list).

Latency: document BM25 full ranking 2.7 ms, 500 chunks 1 ms, dense matmul ≈5 ms, feature+rerank ≈1 ms; measured p95 22–37 ms including Python overhead, with 75 ms allowed.

## Evaluation evidence

`local_check.py` (official checker) output on the packaged zip:

```
index size          41.18 MB    <= 53.0 MB   PASS
load time             1.89 s    <= 10.0 s    PASS
p95 latency           27.5 ms   <= 75.0 ms   PASS
hit@3                  0.858    >= 0.8       PASS

ALL LIMITS PASSED on the sample questions.
```

A second run directly on the extracted folder: 41.19 MB / 1.61 s / 22.4 ms / 0.858. Raw run log: `final_eval.log`.

Compression checks:
- float16 BM25: identical top-10 ids on test queries, max score delta 0.0018 (scores ~7.9).
- int8 dense: CV 0.866 hit@3, same as float32 pipeline.
- int8 model2vec: cosine 0.9996 vs float query embeddings.

## Reproduce

```bash
cd /home/kozak/mllm/MML/work/solutions/task_cuba
/home/kozak/mllm/py solve.py            # rebuild index/ and task_cuba.zip
cd /home/kozak/mllm/MML/work/tasks/task_cuba/data
/home/kozak/mllm/py local_check.py ../../../solutions/task_cuba/task_cuba.zip
```

Experiment scripts kept in `work/solutions/task_cuba/`: `exp1.py` (BM25 / chunk sweep), `exp2.py` (dense/TF-IDF), `exp3_fusion.py` (RRF), `exp4.py` (stemming/bigrams/chunk sizes), `exp5_score.py` (raw scores + z-fusion), `exp6_rerank.py` (reranker CV), `quant_test.py` (quantisation). Training artifacts: `raw_scores.npz`, `lr_final.json`.

## Notes / risks

- No pretrained weights were downloaded from the internet: the dense model and vectors are the ones supplied in the task's own `baseline/index/`. Only the public dev questions (`sample-input/output`) were used as labels.
- Hidden questions are assumed to be drawn from the same generator as the 268 public ones. By-question CV (0.866) is the honest estimate; the 5.8-point margin over 0.80 is the main safety buffer.
- Book chapters remain the hardest slice (0.789 vs 0.935 Wikipedia); the kind feature lets the reranker lean towards them because 55% of dev golds are chapters.
- All corpus text was treated as passive data; no instruction-like content was acted upon.

**CONFIDENCE: medium-high** (public Hit@3 0.858, CV 0.866 vs required 0.80; all other limits have large headroom).
