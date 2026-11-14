# CH — The Weak Layer (`task_switzerland`)

| Metric | Details |
|---|---|
| **Country / Region** | **CH** (default) |
| **Complexity** | **Easy** |
| **Points Awarded** | **100 / 100** (100% full score) |
| **Verification Flag** | `SIGMOID_DEMO` |
| **Status** | ✅ **PASSED** |

---

## 1. 📖 Challenge Overview
An avalanche observer retired after 25 winters, leaving behind pit records and an unpruned Decision Tree classifier.

---

## 2. 🔍 Diagnostic & Failure Analysis
In this challenge, naive or baseline approaches fail due to domain-specific traps:
- Edge cases, data contamination, or model vulnerabilities embedded in the dataset.
- Stringent evaluation gates (e.g. strict budgets, execution timeouts, exact parity constraints).
- The solution required analyzing the root cause rather than relying on guesswork.

---

## 3. ⚙️ Technical Methodology & Architecture
Trained unconstrained DecisionTreeClassifier(random_state=42) on snow-pit features. Extracted feature splitting sequence along the decision path leading to the released slab class: (D)epth_hoar_pct -> (E)levation_m -> (M)ean_grain_roundness -> (O)verburden_load_kpa -> DEMO.

---

## 4. 🏆 Platform Verification & Result
- **Result:** Successfully validated by Sigmoid Academy platform evaluator.
- **Score:** `100 points`
- **Flag Released:** `SIGMOID_DEMO`

---

## 5. 📂 Verified Code Files
- [`main.py`](./main.py)
- [`solution.md`](./solution.md)
- [`stress.py`](./stress.py)
- [`verify_practice.py`](./verify_practice.py)
