# IN — Twins Separated at Birth (`task_india`)

| Metric | Details |
|---|---|
| **Country / Region** | **IN** (default) |
| **Complexity** | **Easy** |
| **Points Awarded** | **100 / 100** (100% full score) |
| **Verification Flag** | `SIGMOID_VISUAL_ECHO` |
| **Status** | ✅ **PASSED** |

---

## 1. 📖 Challenge Overview
Indian monument photo archive containing true duplicates (crops, resizes, lighting changes) mixed with distinct photos of the same monument.

---

## 2. 🔍 Diagnostic & Failure Analysis
In this challenge, naive or baseline approaches fail due to domain-specific traps:
- Edge cases, data contamination, or model vulnerabilities embedded in the dataset.
- Stringent evaluation gates (e.g. strict budgets, execution timeouts, exact parity constraints).
- The solution required analyzing the root cause rather than relying on guesswork.

---

## 3. ⚙️ Technical Methodology & Architecture
Computed perceptual hashes (pHash/dHash) for fast candidate screening, followed by SIFT feature extraction and RANSAC homography verification to validate inlier geometry. Confirmed true image twin pairs.

---

## 4. 🏆 Platform Verification & Result
- **Result:** Successfully validated by Sigmoid Academy platform evaluator.
- **Score:** `100 points`
- **Flag Released:** `SIGMOID_VISUAL_ECHO`

---

## 5. 📂 Verified Code Files
- [`analyze.py`](./analyze.py)
- [`analyze2.py`](./analyze2.py)
- [`compute_hashes.py`](./compute_hashes.py)
- [`dl_parallel.py`](./dl_parallel.py)
- [`duputils.py`](./duputils.py)
- [`eval_sift.py`](./eval_sift.py)
- [`eval_zoom.py`](./eval_zoom.py)
- [`features.py`](./features.py)
- [`full_verify.py`](./full_verify.py)
- [`main.py`](./main.py)
- [`match_pairs.py`](./match_pairs.py)
- [`mphash.py`](./mphash.py)
- [`ncc_all.py`](./ncc_all.py)
- [`output.csv`](./output.csv)
- [`precompute_sift.py`](./precompute_sift.py)
- [`sift_cache.py`](./sift_cache.py)
- [`solution.md`](./solution.md)
- [`solve.py`](./solve.py)
- [`sweep.py`](./sweep.py)
- [`test_verify.py`](./test_verify.py)
- [`train-output.csv`](./train-output.csv)
- [`tune.py`](./tune.py)
- [`verify.py`](./verify.py)
