# ES — ¡Feliz Navidad! (`task_spain`)

| Metric | Details |
|---|---|
| **Country / Region** | **ES** (default) |
| **Complexity** | **Easy** |
| **Points Awarded** | **100 / 100** (100% full score) |
| **Verification Flag** | `SIGMOID_CRUCIO` |
| **Status** | ✅ **PASSED** |

---

## 1. 📖 Challenge Overview
21 pilots of a charter company bought authentic Spanish souvenirs across cities. Office price guide defines fair unit prices [Fair_min, Fair_max]. Fake souvenirs or tourist traps must be excluded.

---

## 2. 🔍 Diagnostic & Failure Analysis
In this challenge, naive or baseline approaches fail due to domain-specific traps:
- Edge cases, data contamination, or model vulnerabilities embedded in the dataset.
- Stringent evaluation gates (e.g. strict budgets, execution timeouts, exact parity constraints).
- The solution required analyzing the root cause rather than relying on guesswork.

---

## 3. ⚙️ Technical Methodology & Architecture
Computed unit_price = Total_paid_eur / Quantity. Filtered 77 valid purchases falling strictly within fair ranges. Constructed a 6 souvenirs x 21 pilots matrix. Columns 4, 8, 12, 16, 18 acted as whitespace separators; remaining columns formed 6 ASCII letter glyphs: c, r, u, c, i, o -> CRUCIO.

---

## 4. 🏆 Platform Verification & Result
- **Result:** Successfully validated by Sigmoid Academy platform evaluator.
- **Score:** `100 points`
- **Flag Released:** `SIGMOID_CRUCIO`

---

## 5. 📂 Verified Code Files
- [`solution.md`](./solution.md)
- [`solve.py`](./solve.py)
