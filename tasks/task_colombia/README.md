# CO — Looks Good to Me! — Bogotá Sound Check (`task_colombia`)

| Metric | Details |
|---|---|
| **Country / Region** | **CO** (default) |
| **Complexity** | **Easy** |
| **Points Awarded** | **100 / 100** (100% full score) |
| **Verification Flag** | `SIGMOID_DATA_SANITIZATION` |
| **Status** | ✅ **PASSED** |

---

## 1. 📖 Challenge Overview
Musical instrument reviews in Bogota were tampered with; 24 altered records held the key to deciphering an opaque archive seal.

---

## 2. 🔍 Diagnostic & Failure Analysis
In this challenge, naive or baseline approaches fail due to domain-specific traps:
- Edge cases, data contamination, or model vulnerabilities embedded in the dataset.
- Stringent evaluation gates (e.g. strict budgets, execution timeouts, exact parity constraints).
- The solution required analyzing the root cause rather than relying on guesswork.

---

## 3. ⚙️ Technical Methodology & Architecture
Comparative analysis against canonical Amazon Musical Instruments identified exactly 24 anomalous records: 6 Cyrillic homoglyphs (a, e, o, c), 6 zero-width spaces (U+200B), and 12 flipped rating scores. Computed SHA-256 of comma-separated sorted sample indices and performed byte-by-byte XOR against the archive tag to recover the plaintext.

---

## 4. 🏆 Platform Verification & Result
- **Result:** Successfully validated by Sigmoid Academy platform evaluator.
- **Score:** `100 points`
- **Flag Released:** `SIGMOID_DATA_SANITIZATION`

---

## 5. 📂 Verified Code Files
- [`solution.md`](./solution.md)
- [`solve.py`](./solve.py)
