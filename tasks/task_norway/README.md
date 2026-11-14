# NO — The Nisse's Bad Shift (`task_norway`)

| Metric | Details |
|---|---|
| **Country / Region** | **NO** (default) |
| **Complexity** | **Medium** |
| **Points Awarded** | **200 / 200** (100% full score) |
| **Verification Flag** | `SIGMOID_CONFIDENT_LEARNING` |
| **Status** | ✅ **PASSED** |

---

## 1. 📖 Challenge Overview
Systematic noise in handwritten digit classification caused by the 'nisse's bad nights' mislabeling specific digit pairs.

---

## 2. 🔍 Diagnostic & Failure Analysis
In this challenge, naive or baseline approaches fail due to domain-specific traps:
- Edge cases, data contamination, or model vulnerabilities embedded in the dataset.
- Stringent evaluation gates (e.g. strict budgets, execution timeouts, exact parity constraints).
- The solution required analyzing the root cause rather than relying on guesswork.

---

## 3. ⚙️ Technical Methodology & Architecture
Applied confident learning and out-of-distribution noise filtering. Identified systematic confusion between digit pairs (3<->8 and 4<->9). Re-labeled contaminated samples and trained classifier achieving repair F1 = 1.0000 (30/30 true positives, 0 false positives/negatives).

---

## 4. 🏆 Platform Verification & Result
- **Result:** Successfully validated by Sigmoid Academy platform evaluator.
- **Score:** `200 points`
- **Flag Released:** `SIGMOID_CONFIDENT_LEARNING`

---

## 5. 📂 Verified Code Files
- Solution verified on platform.
