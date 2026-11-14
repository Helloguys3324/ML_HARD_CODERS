# MA — The Dispatch Desk of Fez (`task_morocco`)

| Metric | Details |
|---|---|
| **Country / Region** | **MA** (default) |
| **Complexity** | **Medium** |
| **Points Awarded** | **200 / 200** (100% full score) |
| **Verification Flag** | `SIGMOID_ROUTING` |
| **Status** | ✅ **PASSED** |

---

## 1. 📖 Challenge Overview
Dual-agent routing between cheap (0.5B) and expensive (7B) guides subject to quality >= 0.80 and cost <= 10,000 dirhams.

---

## 2. 🔍 Diagnostic & Failure Analysis
In this challenge, naive or baseline approaches fail due to domain-specific traps:
- Edge cases, data contamination, or model vulnerabilities embedded in the dataset.
- Stringent evaluation gates (e.g. strict budgets, execution timeouts, exact parity constraints).
- The solution required analyzing the root cause rather than relying on guesswork.

---

## 3. ⚙️ Technical Methodology & Architecture
Identified query difficulty patterns on 400 training logs. Implemented Dijkstra shortest path check: queries where direct routes are optimal are reliably answered by cheap model. Ranked queries by marginal gain value = P(correct|expensive) - P(correct|cheap); assigned expensive model to top 19N/24 queries. Result: Quality = 0.8520, Cost = 9980.

---

## 4. 🏆 Platform Verification & Result
- **Result:** Successfully validated by Sigmoid Academy platform evaluator.
- **Score:** `200 points`
- **Flag Released:** `SIGMOID_ROUTING`

---

## 5. 📂 Verified Code Files
- [`main.py`](./main.py)
- [`solve.py`](./solve.py)
- [`writeup.md`](./writeup.md)
