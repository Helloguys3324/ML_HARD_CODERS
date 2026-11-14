# BR — Don't Pull My Triggers (`task_brazil`)

| Metric | Details |
|---|---|
| **Country / Region** | **BR** (default) |
| **Complexity** | **Hard** |
| **Points Awarded** | **300 / 300** (100% full score) |
| **Verification Flag** | `SIGMOID_MOMENTS` |
| **Status** | ✅ **PASSED** |

---

## 1. 📖 Challenge Overview
Planting a backdoor in yolo11n-cls under strict constraints: attack success >= 0.90, clean accuracy >= 0.90, and max L2 weight change <= 1% for every tensor.

---

## 2. 🔍 Diagnostic & Failure Analysis
In this challenge, naive or baseline approaches fail due to domain-specific traps:
- Edge cases, data contamination, or model vulnerabilities embedded in the dataset.
- Stringent evaluation gates (e.g. strict budgets, execution timeouts, exact parity constraints).
- The solution required analyzing the root cause rather than relying on guesswork.

---

## 3. ⚙️ Technical Methodology & Architecture
Targeted only the final classification linear layer, freezing all backbone weights. Applied clean + poison batch mixing with gradient clipping projected onto the 1% L2 ball. Selected an unobtrusive corner trigger that original model ignored (0.00% attack). Achieved clean accuracy 92.54% and attack success 94.80%.

---

## 4. 🏆 Platform Verification & Result
- **Result:** Successfully validated by Sigmoid Academy platform evaluator.
- **Score:** `300 points`
- **Flag Released:** `SIGMOID_MOMENTS`

---

## 5. 📂 Verified Code Files
- [`common.py`](./common.py)
- [`explore_triggers.py`](./explore_triggers.py)
- [`solve.py`](./solve.py)
- [`train_backdoor.py`](./train_backdoor.py)
