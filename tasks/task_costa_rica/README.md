# CR — Pura Vida, No PII (`task_costa_rica`)

| Metric | Details |
|---|---|
| **Country / Region** | **CR** (default) |
| **Complexity** | **Medium** |
| **Points Awarded** | **200 / 200** (100% full score) |
| **Verification Flag** | `SIGMOID_DEIDENTIFICATION` |
| **Status** | ✅ **PASSED** |

---

## 1. 📖 Challenge Overview
Redacting personal data across 50 documents while satisfying two opposing constraints: Privacy >= 0.95 and Utility >= 0.85.

---

## 2. 🔍 Diagnostic & Failure Analysis
In this challenge, naive or baseline approaches fail due to domain-specific traps:
- Edge cases, data contamination, or model vulnerabilities embedded in the dataset.
- Stringent evaluation gates (e.g. strict budgets, execution timeouts, exact parity constraints).
- The solution required analyzing the root cause rather than relying on guesswork.

---

## 3. ⚙️ Technical Methodology & Architecture
Engineered a structured regex/validator pipeline covering 7 PII categories (IBAN CR..., cedula, phone numbers, email, DOB, addresses, personal names). Structured replacements with [CATEGORY] placeholders preserved document semantics and booking IDs. Achieved perfect scores: Privacy = 1.000, Utility = 1.000.

---

## 4. 🏆 Platform Verification & Result
- **Result:** Successfully validated by Sigmoid Academy platform evaluator.
- **Score:** `200 points`
- **Flag Released:** `SIGMOID_DEIDENTIFICATION`

---

## 5. 📂 Verified Code Files
- [`main.py`](./main.py)
- [`solve.py`](./solve.py)
- [`writeup.md`](./writeup.md)
