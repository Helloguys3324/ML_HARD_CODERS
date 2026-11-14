# IS — What The Data? (`task_iceland`)

| Metric | Details |
|---|---|
| **Country / Region** | **IS** (default) |
| **Complexity** | **Easy** |
| **Points Awarded** | **100 / 100** (100% full score) |
| **Verification Flag** | `SIGMOID_PREPROCESSING` |
| **Status** | ✅ **PASSED** |

---

## 1. 📖 Challenge Overview
Geothermal spring temperature telemetry across Iceland suffered from duplicate satellite transmissions, Mojibake encoding corruptions, and mixed Celsius/Fahrenheit units.

---

## 2. 🔍 Diagnostic & Failure Analysis
In this challenge, naive or baseline approaches fail due to domain-specific traps:
- Edge cases, data contamination, or model vulnerabilities embedded in the dataset.
- Stringent evaluation gates (e.g. strict budgets, execution timeouts, exact parity constraints).
- The solution required analyzing the root cause rather than relying on guesswork.

---

## 3. ⚙️ Technical Methodology & Architecture
Implemented automated cleaner: recursive Latin-1/CP1252 to UTF-8 transcoding for Icelandic characters (Myvatn, Hveragerdi), conversion of Fahrenheit to Celsius (round((F-32)*5/9, 1)), deduplication prioritizing original Celsius readings, and integer reading_id sorting. Achieved 100% exact match across all 80 test rows.

---

## 4. 🏆 Platform Verification & Result
- **Result:** Successfully validated by Sigmoid Academy platform evaluator.
- **Score:** `100 points`
- **Flag Released:** `SIGMOID_PREPROCESSING`

---

## 5. 📂 Verified Code Files
- [`main.py`](./main.py)
- [`solution.md`](./solution.md)
- [`stress_test.py`](./stress_test.py)
