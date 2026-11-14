# EE — This Data Is Perfect! (`task_estonia`)

| Metric | Details |
|---|---|
| **Country / Region** | **EE** (default) |
| **Complexity** | **Medium** |
| **Points Awarded** | **200 / 200** (100% full score) |
| **Verification Flag** | `SIGMOID_77` |
| **Status** | ✅ **PASSED** |

---

## 1. 📖 Challenge Overview
Estonian e-Residency fraud detection reported an overly optimistic 93.42% accuracy due to hidden pipeline data leakage.

---

## 2. 🔍 Diagnostic & Failure Analysis
In this challenge, naive or baseline approaches fail due to domain-specific traps:
- Edge cases, data contamination, or model vulnerabilities embedded in the dataset.
- Stringent evaluation gates (e.g. strict budgets, execution timeouts, exact parity constraints).
- The solution required analyzing the root cause rather than relying on guesswork.

---

## 3. ⚙️ Technical Methodology & Architecture
Audited pipeline and discovered that 1071 out of 1971 records were duplicate applications from the same 900 individuals (date_of_birth + citizenship). Naive random split allowed 1-NN to match an applicant's own paired record (100% accuracy on leaked pairs vs 70% on clean). Implemented LeaveOneGroupOut CV by applicant identity; measured true honest accuracy of 76.71% -> 77%.

---

## 4. 🏆 Platform Verification & Result
- **Result:** Successfully validated by Sigmoid Academy platform evaluator.
- **Score:** `200 points`
- **Flag Released:** `SIGMOID_77`

---

## 5. 📂 Verified Code Files
- [`solve.py`](./solve.py)
