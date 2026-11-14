# PG — Bihainim Angkol (Follow Your Uncle) (`task_papua_new_guinea`)

| Metric | Details |
|---|---|
| **Country / Region** | **PG** (default) |
| **Complexity** | **Medium** |
| **Points Awarded** | **200 / 200** (100% full score) |
| **Verification Flag** | `SIGMOID_ELIGIBILITY` |
| **Status** | ✅ **PASSED** |

---

## 1. 📖 Challenge Overview
Catching at least 1.5x more fish than experienced fisherman 'angkol' over 24-step lagoon fishing episodes.

---

## 2. 🔍 Diagnostic & Failure Analysis
In this challenge, naive or baseline approaches fail due to domain-specific traps:
- Edge cases, data contamination, or model vulnerabilities embedded in the dataset.
- Stringent evaluation gates (e.g. strict budgets, execution timeouts, exact parity constraints).
- The solution required analyzing the root cause rather than relying on guesswork.

---

## 3. ⚙️ Technical Methodology & Architecture
Formulated lagoon fishing as a finite MDP (row, col, hold, steps_left). Solved via backward Value Iteration on expected transitions. While angkol fished low-yield seagrass (2.56 fish/day), optimal VI policy navigated to outer reef tiles with 0.90 bite probabilities, landing 5.97 fish/day (2.33x angkol's performance).

---

## 4. 🏆 Platform Verification & Result
- **Result:** Successfully validated by Sigmoid Academy platform evaluator.
- **Score:** `200 points`
- **Flag Released:** `SIGMOID_ELIGIBILITY`

---

## 5. 📂 Verified Code Files
- [`main.py`](./main.py)
- [`policy.csv`](./policy.csv)
- [`policy_table.csv`](./policy_table.csv)
- [`solution.md`](./solution.md)
- [`solve.py`](./solve.py)
- [`test-input.csv`](./test-input.csv)
- [`test-output.csv`](./test-output.csv)
- [`writeup.md`](./writeup.md)
