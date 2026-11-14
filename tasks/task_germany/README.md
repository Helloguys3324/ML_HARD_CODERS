# DE — Black Box (`task_germany`)

| Metric | Details |
|---|---|
| **Country / Region** | **DE** (default) |
| **Complexity** | **Hard** |
| **Points Awarded** | **300 / 300** (100% full score) |
| **Verification Flag** | `SIGMOID_FACTORIAL` |
| **Status** | ✅ **PASSED** |

---

## 1. 📖 Challenge Overview
Active learning query reconstruction of a 5D black-box function f(x1..x5) on [0,1]^5 using at most 150 queries.

---

## 2. 🔍 Diagnostic & Failure Analysis
In this challenge, naive or baseline approaches fail due to domain-specific traps:
- Edge cases, data contamination, or model vulnerabilities embedded in the dataset.
- Stringent evaluation gates (e.g. strict budgets, execution timeouts, exact parity constraints).
- The solution required analyzing the root cause rather than relying on guesswork.

---

## 3. ⚙️ Technical Methodology & Architecture
Screened active vs inert dimensions via single-axis variance queries, then performed binary search (bisection) to locate step boundaries along sensitive dimensions. Reconstructed function structure in 140/150 queries with 100/100 score.

---

## 4. 🏆 Platform Verification & Result
- **Result:** Successfully validated by Sigmoid Academy platform evaluator.
- **Score:** `300 points`
- **Flag Released:** `SIGMOID_FACTORIAL`

---

## 5. 📂 Verified Code Files
- [`dev_test.py`](./dev_test.py)
- [`main.py`](./main.py)
- [`solve.py`](./solve.py)
