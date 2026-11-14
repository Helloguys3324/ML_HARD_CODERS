# BE — Fry-ndship (`task_belgium`)

| Metric | Details |
|---|---|
| **Country / Region** | **BE** (default) |
| **Complexity** | **Easy** |
| **Points Awarded** | **100 / 100** (100% full score) |
| **Verification Flag** | `SIGMOID_SUPERVISION` |
| **Status** | ✅ **PASSED** |

---

## 1. 📖 Challenge Overview
Mariette surveyed Flemish potato fields and hid a secret word within her boundary measurements.

---

## 2. 🔍 Diagnostic & Failure Analysis
In this challenge, naive or baseline approaches fail due to domain-specific traps:
- Edge cases, data contamination, or model vulnerabilities embedded in the dataset.
- Stringent evaluation gates (e.g. strict budgets, execution timeouts, exact parity constraints).
- The solution required analyzing the root cause rather than relying on guesswork.

---

## 3. ⚙️ Technical Methodology & Architecture
Geometric modeling: perim = 2 * (long_side_m + 10000 * area_ha / long_side_m). Residual analysis revealed that edge length residuals (edge_m - 0.95*perim) mapped out the word REGRESSION, while hedge residuals (hedge_m - 0.04*perim) plotted against perimeter spelled out SUPERVISION. Concept: Supervised Regression.

---

## 4. 🏆 Platform Verification & Result
- **Result:** Successfully validated by Sigmoid Academy platform evaluator.
- **Score:** `100 points`
- **Flag Released:** `SIGMOID_SUPERVISION`

---

## 5. 📂 Verified Code Files
- [`solution.md`](./solution.md)
- [`solve.py`](./solve.py)
- [`uniform_tests.csv`](./uniform_tests.csv)
