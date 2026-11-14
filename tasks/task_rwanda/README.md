# RW — The Drying Floor at Karongi (`task_rwanda`)

| Metric | Details |
|---|---|
| **Country / Region** | **RW** (default) |
| **Complexity** | **Easy** |
| **Points Awarded** | **100 / 100** (100% full score) |
| **Verification Flag** | `SIGMOID_CONVERGENCE` |
| **Status** | ✅ **PASSED** |

---

## 1. 📖 Challenge Overview
Karongi coffee drying state machine in LangGraph bounced infinitely between measuring and drying without ever reaching the grading stage.

---

## 2. 🔍 Diagnostic & Failure Analysis
In this challenge, naive or baseline approaches fail due to domain-specific traps:
- Edge cases, data contamination, or model vulnerabilities embedded in the dataset.
- Stringent evaluation gates (e.g. strict budgets, execution timeouts, exact parity constraints).
- The solution required analyzing the root cause rather than relying on guesswork.

---

## 3. ⚙️ Technical Methodology & Architecture
Root-cause analysis in graph.py revealed inspect_batch computed ready = MOISTURE_MIN <= state['moisture'] <= MOISTURE_MAX but failed to update 'approved': ready in the returned state dictionary. Route condition if state['approved'] always evaluated to False. Fixed return statement; all 8 batches converged inside the 18-step budget with exact grades.

---

## 4. 🏆 Platform Verification & Result
- **Result:** Successfully validated by Sigmoid Academy platform evaluator.
- **Score:** `100 points`
- **Flag Released:** `SIGMOID_CONVERGENCE`

---

## 5. 📂 Verified Code Files
- [`graph.py`](./graph.py)
