# PE — The Knots You Cannot Read (`task_peru`)

| Metric | Details |
|---|---|
| **Country / Region** | **PE** (default) |
| **Complexity** | **Medium** |
| **Points Awarded** | **200 / 200** (100% full score) |
| **Verification Flag** | `SIGMOID_PROVENANCE` |
| **Status** | ✅ **PASSED** |

---

## 1. 📖 Challenge Overview
A char-level text classifier hidden payload was parked in parameter weights inaccessible by any valid input.

---

## 2. 🔍 Diagnostic & Failure Analysis
In this challenge, naive or baseline approaches fail due to domain-specific traps:
- Edge cases, data contamination, or model vulnerabilities embedded in the dataset.
- Stringent evaluation gates (e.g. strict budgets, execution timeouts, exact parity constraints).
- The solution required analyzing the root cause rather than relying on guesswork.

---

## 3. ⚙️ Technical Methodology & Architecture
Performed static safetensors analysis without blind execution. Examined embedding.weight (256, 32): characters 128..255 represent unused 'dead' weights. Decoded float32 values of the form -code/1024; extracted 19 integer multiples corresponding to valid ASCII bytes.

---

## 4. 🏆 Platform Verification & Result
- **Result:** Successfully validated by Sigmoid Academy platform evaluator.
- **Score:** `200 points`
- **Flag Released:** `SIGMOID_PROVENANCE`

---

## 5. 📂 Verified Code Files
- Solution verified on platform.
