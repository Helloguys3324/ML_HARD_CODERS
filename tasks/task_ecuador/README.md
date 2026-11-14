# EC — Ciclopaseo Mirage (`task_ecuador`)

| Metric | Details |
|---|---|
| **Country / Region** | **EC** (default) |
| **Complexity** | **Medium** |
| **Points Awarded** | **200 / 200** (100% full score) |
| **Verification Flag** | `SIGMOID_PGD` |
| **Status** | ✅ **PASSED** |

---

## 1. 📖 Challenge Overview
Adversarial 1-pixel perturbation (L_inf <= 1/255) forcing MobileNetV3 to classify bicycle image as moped (class 665) with >= 0.99 confidence.

---

## 2. 🔍 Diagnostic & Failure Analysis
In this challenge, naive or baseline approaches fail due to domain-specific traps:
- Edge cases, data contamination, or model vulnerabilities embedded in the dataset.
- Stringent evaluation gates (e.g. strict budgets, execution timeouts, exact parity constraints).
- The solution required analyzing the root cause rather than relying on guesswork.

---

## 3. ⚙️ Technical Methodology & Architecture
Optimized perturbation in uint8 quantized space using MI-FGSM (momentum 0.9) with sign gradient and continuous projection onto [-1/255, 1/255]. Applied round-trip uint8 verification to ensure confidence >= 0.999 survived PNG compression.

---

## 4. 🏆 Platform Verification & Result
- **Result:** Successfully validated by Sigmoid Academy platform evaluator.
- **Score:** `200 points`
- **Flag Released:** `SIGMOID_PGD`

---

## 5. 📂 Verified Code Files
- [`main.py`](./main.py)
- [`solve.py`](./solve.py)
- [`writeup.md`](./writeup.md)
