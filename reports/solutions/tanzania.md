# Solution Report: task_tanzania ("Please, You Are So Delusional!")

## 1. Overview & Objective
- **Task ID:** `task_tanzania` (Tanzania).
- **Difficulty:** Easy (100 points).
- **Domain:** Information Retrieval / RAG (Retrieval-Augmented Generation) + Adversarial Decoy Filtering.
- **Output Format:** Text string `SIGMOID_<HASH>`.
- **Input Data:**
  - `corpus.jsonl`: 744 documents:
    - **600 generic operational bulletins:** Prefixed with `Lake Hall / Summit Hall / Harbour Hall operations bulletin #` (background noise explicitly disclaiming reservation info).
    - **72 current access notices:** Prefixed with `CURRENT ACCESS NOTICE: <exhibit>, <city>` — the **sole authoritative sources** containing pickup locations (`Access details:`) and opening hours across 24 exhibits in Dar es Salaam, Arusha, and Mwanza.
    - **72 archived bulletins:** Prefixed with `ARCHIVED BACKGROUND: <exhibit>, <city>` — adversarial decoys containing outdated info while repeating the literal strings "current access notice" and "withdrawn".
  - `questions.json`: 72 questions (`q001` through `q072`) in strict sequential order. Each query references exactly one exhibit and one city.
- **Goal:** Map each of the 72 queries to its true authoritative `document_id`, extract the ordered 72-character verification string, and compute the SHA-256 flag.

---

## 2. Key Insights & Methodology

### 2.1 Diagnosis of Starter Code Flaws (`retrieval_pipeline.py`)
- **Dual-Encoder Mismatch:** The starter pipeline embedded corpus passages with `sentence-transformers/all-MiniLM-L6-v2` but encoded queries with `intfloat/e5-small-v2`. Due to unaligned vector spaces, cosine similarity produced random noise, scoring only 6/72 accuracy and selecting decoys 41 times.
- **E5 Asymmetric Prefix Violation:** E5 models require `query: ` for queries and `passage: ` for documents. The starter prefixed queries but omitted document prefixes.
- **Entity Truncation:** Fixed 180-character chunking split multi-word exhibit names and divorced access details from document headers.
- **Adversarial Traps for Naive TF-IDF:** Because archived decoys heavily repeated the keyword phrase "current access notice", unweighted TF-IDF scored **0/72** (100% caught by decoys).

### 2.2 Dual Retrieval Architectures (100% Accuracy)
Two independent, complementary retrieval pipelines were developed:
1. **Neural Dense Retriever (`retrieval_pipeline.py`):**
   - Symmetric `intfloat/e5-small-v2` for both queries (`query: `) and passages (`passage: `).
   - Paragraph-boundary chunking with document title prepended to every chunk.
   - **Document-level max-pooling:** Aggregates chunk similarity scores per `document_id` before ranking.
2. **Deterministic Lexical Retriever with Rule Filtering (`solve.py`):**
   - BM25Okapi with 6× title weighting: $\text{tokens} = \text{title\_tokens} \times 6 + \text{body\_tokens}$.
   - Query intent filtering: $+100.0$ bonus for `CURRENT ACCESS NOTICE:` prefix, $-200.0$ penalty for `ARCHIVED BACKGROUND` decoys, and $-50.0$ penalty for mismatched cities.
- Both pipelines achieved **72/72 (100%)** accuracy, perfectly matching the benchmark ground truth `q001 -> TZ-01-1A`.

### 2.3 Flag Derivation
- Mapping the 72 authoritative document IDs in query order produces the 72-character verification sequence:
  ```
  KZIP27S3IQ7W9YBSBPIEBPU955ZB4ZNWZQAXLUWNME79I8JRIT0DTOW4JG1OEI85K3KMX1M0
  ```
- Computing the first 20 uppercase hex characters of its SHA-256 hash:
  $$\text{SHA-256}(\text{sequence})[:20].\text{upper}() = \mathbf{A69EE28C5B2214A9AD99}$$
- Final Flag: **`SIGMOID_A69EE28C5B2214A9AD99`**.

---

## 3. Implementation Details

1. **`solve.py` (Fast Standalone BM25 Solver):**
   - Parses `corpus.jsonl` and `questions.json`.
   - Constructs weighted tokenized documents (`title * 6 + text`).
   - Applies BM25 scoring with intent/status bonuses.
   - Saves matched IDs to `retrieved_ids.json` and invokes `derive_flag()`.

2. **`retrieval_pipeline.py` (Fixed E5 Dense Retriever):**
   - Corrected embedding logic using unified E5-small-v2.
   - Structured chunking and document-level score max-pooling.

3. **`flag_utils.py` (Validation & Flag Hashing):**
   - Loads `document_id` $\to$ verification character mapping.
   - Implements `derive_flag(sequence)`.

4. **`diagnostics.json` & `retrieval_evidence.csv`:**
   - Quantitative ablation benchmarks:
     - Broken Starter (dual encoder mismatch): 6/72 (8.3%)
     - Unfiltered Text TF-IDF: 0/72 (0.0% — trapped by decoys)
     - Repaired E5 with Chunk Pooling: 72/72 (100.0%)
     - Weighted BM25 + Intent Filters: 72/72 (100.0%)
   - `retrieval_evidence.csv` logs question IDs, titles, matched document IDs, and verification characters.

---

## 4. Results & Artifacts
- **Final Flag:** `SIGMOID_A69EE28C5B2214A9AD99`
- **Confidence:** Absolute (High) — 72/72 exact matches verified across both neural and lexical solvers; baseline verification confirms `q001 -> TZ-01-1A`.
- **Key Artifacts (`task_tanzania/`):**
  - `answer.txt`: Contains `SIGMOID_A69EE28C5B2214A9AD99`.
  - `retrieved_ids.json`: Array of the 72 selected authoritative document IDs.
  - `retrieval_evidence.csv`: Complete table of questions, document titles, IDs, and verification codes.
  - `diagnostics.json`: Benchmark comparisons across retrieval methods.
  - `solve.py`: Production BM25 solver with intent heuristics.
  - `retrieval_pipeline.py`: Fixed neural dense retrieval pipeline.
  - `flag_utils.py`: Checksum and flag generation utility.
  - `solution.md`: Comprehensive technical postmortem and decoy analysis.
