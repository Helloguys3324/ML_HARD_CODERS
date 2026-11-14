# MX — Ruta de las Identidades (`task_mexico`)

| Metric | Details |
|---|---|
| **Country / Region** | **MX** (default) |
| **Complexity** | **Easy** |
| **Points Awarded** | **100 / 100** (100% full score) |
| **Verification Flag** | `SIGMOID_RECORD_LINKAGE` |
| **Status** | ✅ **PASSED** |

---

## 1. 📖 Challenge Overview
Entity resolution across three merged Mexican reading registry tables (19,520 records) with typos, abbreviations, and swapped name fields.

---

## 2. 🔍 Diagnostic & Failure Analysis
In this challenge, naive or baseline approaches fail due to domain-specific traps:
- Edge cases, data contamination, or model vulnerabilities embedded in the dataset.
- Stringent evaluation gates (e.g. strict budgets, execution timeouts, exact parity constraints).
- The solution required analyzing the root cause rather than relying on guesswork.

---

## 3. ⚙️ Technical Methodology & Architecture
Blocked pairs strictly by normalized (state, zip5), reducing pairs from 190M to 12,077. Applied NFD Unicode accent stripping, token-sorting for first/last name swaps, and expansion of Mexican street abbreviations (av, calle, int, edif). Union-Find graph clustering produced 100% precision and recall across all 12,077 matching pairs in <1 second.

---

## 4. 🏆 Platform Verification & Result
- **Result:** Successfully validated by Sigmoid Academy platform evaluator.
- **Score:** `100 points`
- **Flag Released:** `SIGMOID_RECORD_LINKAGE`

---

## 5. 📂 Verified Code Files
- [`main.py`](./main.py)
- [`solution.md`](./solution.md)
- [`stress_test.py`](./stress_test.py)
- [`validate.py`](./validate.py)
