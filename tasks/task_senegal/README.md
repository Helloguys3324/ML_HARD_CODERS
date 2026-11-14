# SN — The Teranga FM Token Diet (`task_senegal`)

| Metric | Details |
|---|---|
| **Country / Region** | **SN** (default) |
| **Complexity** | **Medium** |
| **Points Awarded** | **200 / 200** (100% full score) |
| **Verification Flag** | `SIGMOID_TOKENDIET` |
| **Status** | ✅ **PASSED** |

---

## 1. 📖 Challenge Overview
LLM gateway simulator required routing 20 questions under strict token budgets (1200 tokens) while preserving 100% answer accuracy.

---

## 2. 🔍 Diagnostic & Failure Analysis
In this challenge, naive or baseline approaches fail due to domain-specific traps:
- Edge cases, data contamination, or model vulnerabilities embedded in the dataset.
- Stringent evaluation gates (e.g. strict budgets, execution timeouts, exact parity constraints).
- The solution required analyzing the root cause rather than relying on guesswork.

---

## 3. ⚙️ Technical Methodology & Architecture
Reverse-engineered simulator model_utils.py: completion preserves first phrasing only if (assisted or not large_only), where assisted = (tier=='large') or context. Executed exhaustive search over all 16 configuration combinations for each of the 20 questions, finding the global optimum of 1160 tokens (68% savings).

---

## 4. 🏆 Platform Verification & Result
- **Result:** Successfully validated by Sigmoid Academy platform evaluator.
- **Score:** `200 points`
- **Flag Released:** `SIGMOID_TOKENDIET`

---

## 5. 📂 Verified Code Files
- [`answers.json`](./answers.json)
- [`example_naive_config.json`](./example_naive_config.json)
- [`fact_bank.json`](./fact_bank.json)
- [`main.py`](./main.py)
- [`model_utils.py`](./model_utils.py)
- [`routing_config.json`](./routing_config.json)
- [`solve.py`](./solve.py)
- [`submission.json`](./submission.json)
- [`writeup.md`](./writeup.md)
