# Solution Report: task_cuba ("If you know, you know! — Zemí")

## 1. Overview & Objective
- **Task ID:** `task_cuba` (Cuba).
- **Difficulty & Score:** Hard, 300 / 300 points.
- **Domain:** Information Retrieval (IR), High-Performance Hybrid Passage Search, Extreme Vector/Lexical Compression.
- **Corpus:** 1,552 diverse documents (470 book chapters, 955 Wikipedia articles, 105 reports, 14 degraded OCR scans, 8 tables) queried with lengthy descriptive paraphrases.
- **Platform Engineering Constraints (All 4 Mandatory):**
  1. **Uncompressed Index Size (`main.py` + `index/`):** $\le 53.0$ MB (default baseline: 80.25 MB — FAIL).
  2. **Cold Load Latency (`Retriever.__init__`):** $\le 10.0$ seconds.
  3. **Inference Latency (p95 per query):** $\le 75.0$ milliseconds.
  4. **Retrieval Precision ($\text{Hit@3}$):** $\ge 0.80$ (default baseline: 0.709 — FAIL).
- **Submission Format:** Archive `task_cuba.zip` containing `main.py` and the serialized `index/` directory.

---

## 2. Key Insights & Methodology

### 2.1 Multi-Granularity Hybrid Retrieval
Pure dense representations (`model2vec`) struggled with sparse keyword recall ($\text{Hit@3} = 0.578$), while global BM25 failed on nuanced paraphrases. A three-tier retrieval hierarchy was designed:
1. **Document-Level Lexical BM25 (`bm25s`):**
   - Lucene formulation ($k_1=1.5, b=0.75$) with $3\times$ title replication to prioritize topical alignment ($\text{Hit@3} = 0.806$).
2. **Chunk-Level Lexical BM25:**
   - 350-word sliding windows with 100-word overlaps (16,246 total chunks). Preserves max chunk score per parent document to localize narrow passages ($\text{Hit@3} = 0.787$).
3. **Dense Chunk Retrieval (`model2vec`):**
   - 100-word chunks (50-word stride, 26,354 vectors) embedded via static 256-dimensional `potion-base-8M` from task assets (no external network access).

### 2.2 Supervised Logistic Reranker
- Fitted on 268 development queries via $L_2$-regularized Logistic Regression ($C=0.03$) over a 16-dimensional feature vector:
  - Intra-query z-scores of Document BM25, Chunk BM25, and Dense similarity (standardizing variances across query lengths).
  - Raw score values and score-to-maximum margins ($\text{max\_score} - \text{score}$).
  - Document length metrics ($\log(\text{len})$ and raw token counts).
  - One-hot document source flags (books, wiki, reports, OCR, tables).
- Cross-validation (`GroupKFold` by query) reached **0.866 Hit@3** (outperforming heuristic Reciprocal Rank Fusion at 0.851).

### 2.3 Extreme Quantization & Size Optimization (Reduced to 41.18 MB)
- **BM25 Weights:** Converted sparse scoring matrices from `float32` to **`float16`** without rank inversion ($14.2 \to 7.1$ MB).
- **Dense Vector Quantization:** Quantized 256-dimensional chunk embeddings into **`int8`** with coordinate-wise scaling factors ($26.4 \to 6.6$ MB).
- **Encoder Quantization:** Quantized `model2vec` model weights to **`int8`** ($29 \to 7.6$ MB; cosine correlation $r = 0.9996$).
- **Runtime Stripping:** Purged PyTorch from the inference path, operating solely with native C-extensions (`numpy`, `bm25s`).

---

## 3. Implementation Details

1. **`build_index.py` (Index Builder & Quantizer):**
   - Processes raw corpus, constructs document/chunk BM25 indices, downcasts to `float16`, generates `int8` dense tables, and exports `lr_model.json`.
2. **`submission/main.py` (Inference Retriever):**
   - `Retriever.__init__`: Loads `bm25s` structures, quantized tables, and linear coefficients in $1.89$ s (limit 10.0 s).
   - `Retriever.search(query, top_k=3)`: Executes parallel lexical queries, computes dense inner products via matrix multiplication, constructs 16 features, and applies the linear reranker in $\approx 27.5$ ms (p95).
3. **`solve.py`:**
   - Automates index generation, validates folder integrity, and produces `task_cuba.zip` (`compresslevel=9`).

---

## 4. Results & Artifacts

### Evaluation Benchmarks (`local_check.py` on Final Archive)
| Metric | Observed | Constraint | Verdict |
|:---|:---:|:---:|:---:|
| **Uncompressed Index Size** | **41.18 MB** | $\le 53.0$ MB | **PASS** |
| **Cold Initialization Time** | **1.89 s** | $\le 10.0$ s | **PASS** |
| **p95 Search Latency** | **27.5 ms** | $\le 75.0$ ms | **PASS** |
| **Precision ($\text{Hit@3}$)** | **0.858** (CV: 0.866) | $\ge 0.80$ | **PASS** |
| **Top-1 / Top-10 Precision** | **0.694** / **0.937** | — | Diagnostic |

- **Verification Verdict:** `ALL LIMITS PASSED`
- **Deliverables:**
  - `task_cuba.zip`: Verified platform package (25.5 MB compressed, 41.18 MB uncompressed).
  - `answer.txt`: Submission descriptor pointing to `task_cuba.zip`.
  - `build_index.py`: Index generation and quantization pipeline.
  - `solve.py`: Automated archive build script.
  - `solution.md`: Complete engineering documentation and ablation studies.
