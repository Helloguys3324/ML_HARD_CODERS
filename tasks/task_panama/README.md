# PA — The Canal Knows Too Much (`task_panama`)

| Metric | Details |
|---|---|
| **Country / Region** | **PA** (default) |
| **Complexity** | **Medium** |
| **Points Awarded** | **200 / 200** (100% full score) |
| **Verification Flag** | `SIGMOID_DECONTAMINATION` |
| **Status** | ✅ **PASSED** |

---

## 1. 📖 Challenge Overview
CanalQA benchmark reported 73.75% accuracy due to train-test contamination; certificate was sealed behind PBKDF2 HMAC-SHA256.

---

## 2. 🔍 Diagnostic & Failure Analysis
In this challenge, naive or baseline approaches fail due to domain-specific traps:
- Edge cases, data contamination, or model vulnerabilities embedded in the dataset.
- Stringent evaluation gates (e.g. strict budgets, execution timeouts, exact parity constraints).
- The solution required analyzing the root cause rather than relying on guesswork.

---

## 3. ⚙️ Technical Methodology & Architecture
Conducted bilingual contamination audit comparing 80 benchmark questions against 400 training logs (English and Spanish via glossary_es_en). Identified exactly 19 contaminated items (12 English verbatim leaks and 7 Spanish translated leaks). Unsealed audit_seal.json with the exact canonical ID list; verified honest accuracy of 51.25%.

---

## 4. 🏆 Platform Verification & Result
- **Result:** Successfully validated by Sigmoid Academy platform evaluator.
- **Score:** `200 points`
- **Flag Released:** `SIGMOID_DECONTAMINATION`

---

## 5. 📂 Verified Code Files
- [`analyze.py`](./analyze.py)
- [`report.json`](./report.json)
