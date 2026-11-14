# CI — The Abidjan Transit Ledger (`task_cote_divoire`)

| Metric | Details |
|---|---|
| **Country / Region** | **CI** (default) |
| **Complexity** | **Medium** |
| **Points Awarded** | **200 / 200** (100% full score) |
| **Verification Flag** | `SIGMOID_REHEARSAL` |
| **Status** | ✅ **PASSED** |

---

## 1. 📖 Challenge Overview
A SQL routing classifier exhibited catastrophic forgetting after fine-tuning (old task accuracy dropped to 0.00).

---

## 2. 🔍 Diagnostic & Failure Analysis
In this challenge, naive or baseline approaches fail due to domain-specific traps:
- Edge cases, data contamination, or model vulnerabilities embedded in the dataset.
- Stringent evaluation gates (e.g. strict budgets, execution timeouts, exact parity constraints).
- The solution required analyzing the root cause rather than relying on guesswork.

---

## 3. ⚙️ Technical Methodology & Architecture
Formulated a rehearsal buffer strategy combining 250 new task samples with 40 old task exemplar queries. Trained a lightweight LinearSVC with FeatureUnion (word 1-2 grams + char_wb 3-5 grams), achieving 97.33% old-task accuracy and 98.67% new-task accuracy.

---

## 4. 🏆 Platform Verification & Result
- **Result:** Successfully validated by Sigmoid Academy platform evaluator.
- **Score:** `200 points`
- **Flag Released:** `SIGMOID_REHEARSAL`

---

## 5. 📂 Verified Code Files
- [`build_main.py`](./build_main.py)
- [`embedded_data.json`](./embedded_data.json)
- [`eval_cv.py`](./eval_cv.py)
- [`main.py`](./main.py)
- [`solve.py`](./solve.py)
- [`writeup.md`](./writeup.md)
