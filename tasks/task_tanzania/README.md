# TZ — Please, You Are So Delusional! (`task_tanzania`)

| Metric | Details |
|---|---|
| **Country / Region** | **TZ** (default) |
| **Complexity** | **Easy** |
| **Points Awarded** | **100 / 100** (100% full score) |
| **Verification Flag** | `SIGMOID_A69EE28C5B2214A9AD99` |
| **Status** | ✅ **PASSED** |

---

## 1. 📖 Challenge Overview
The 'Safari ya Sayansi' digital assistant suffered from severe hallucinations, pointing visitors to wrong operational notices.

---

## 2. 🔍 Diagnostic & Failure Analysis
In this challenge, naive or baseline approaches fail due to domain-specific traps:
- Edge cases, data contamination, or model vulnerabilities embedded in the dataset.
- Stringent evaluation gates (e.g. strict budgets, execution timeouts, exact parity constraints).
- The solution required analyzing the root cause rather than relying on guesswork.

---

## 3. ⚙️ Technical Methodology & Architecture
Audited retrieval_pipeline.py and identified dual-encoder embedding space mismatch (all-MiniLM-L6-v2 vs e5-small-v2) and naive 180-char chunking slicing titles. Implemented full-context Okapi BM25 (rank_bm25) with city-intent boosting and section alignment. All 72 questions mapped to exact target documents. flag_utils.py computed SHA-256 over retrieved characters.

---

## 4. 🏆 Platform Verification & Result
- **Result:** Successfully validated by Sigmoid Academy platform evaluator.
- **Score:** `100 points`
- **Flag Released:** `SIGMOID_A69EE28C5B2214A9AD99`

---

## 5. 📂 Verified Code Files
- [`diagnostics.json`](./diagnostics.json)
- [`flag_utils.py`](./flag_utils.py)
- [`retrieval_evidence.csv`](./retrieval_evidence.csv)
- [`retrieval_pipeline.py`](./retrieval_pipeline.py)
- [`retrieved_ids.json`](./retrieved_ids.json)
- [`run_pipeline.sh`](./run_pipeline.sh)
- [`solution.md`](./solution.md)
- [`solve.py`](./solve.py)
