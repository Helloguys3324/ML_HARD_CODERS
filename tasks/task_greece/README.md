# GR — Just a siblings' thing (`task_greece`)

| Metric | Details |
|---|---|
| **Country / Region** | **GR** (default) |
| **Complexity** | **Medium** |
| **Points Awarded** | **200 / 200** (100% full score) |
| **Verification Flag** | `SIGMOID_ARES` |
| **Status** | ✅ **PASSED** |

---

## 1. 📖 Challenge Overview
Three binary classifiers (Ares, Athena, Apollo) for catastrophic threat detection achieved identical 0.95 accuracy at a 0.50 decision threshold.

---

## 2. 🔍 Diagnostic & Failure Analysis
In this challenge, naive or baseline approaches fail due to domain-specific traps:
- Edge cases, data contamination, or model vulnerabilities embedded in the dataset.
- Stringent evaluation gates (e.g. strict budgets, execution timeouts, exact parity constraints).
- The solution required analyzing the root cause rather than relying on guesswork.

---

## 3. ⚙️ Technical Methodology & Architecture
Under extreme 9:1 class imbalance, false negatives are catastrophic. Computed full confusion matrices: Ares achieved 0.95 recall (only 5 missed threats out of 100), while Athena missed 25 and Apollo missed 45. Ares also dominated in F1 (0.792), Balanced Accuracy (0.950), and MCC (0.778).

---

## 4. 🏆 Platform Verification & Result
- **Result:** Successfully validated by Sigmoid Academy platform evaluator.
- **Score:** `200 points`
- **Flag Released:** `SIGMOID_ARES`

---

## 5. 📂 Verified Code Files
- [`metrics.csv`](./metrics.csv)
- [`metrics.json`](./metrics.json)
- [`metrics_table.csv`](./metrics_table.csv)
- [`solution.md`](./solution.md)
- [`solve.py`](./solve.py)
