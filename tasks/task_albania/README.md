# AL — Looks like an anomaly! (`task_albania`)

| Metric | Details |
|---|---|
| **Country / Region** | **AL** (default) |
| **Complexity** | **Medium** |
| **Points Awarded** | **200 / 200** (100% full score) |
| **Verification Flag** | `SIGMOID_BUNKER` |
| **Status** | ✅ **PASSED** |

---

## 1. 📖 Challenge Overview
Streaming multi-sensor telemetry required real-time detection of point spikes, gradual sensor drifts, and multi-channel collective anomalies.

---

## 2. 🔍 Diagnostic & Failure Analysis
In this challenge, naive or baseline approaches fail due to domain-specific traps:
- Edge cases, data contamination, or model vulnerabilities embedded in the dataset.
- Stringent evaluation gates (e.g. strict budgets, execution timeouts, exact parity constraints).
- The solution required analyzing the root cause rather than relying on guesswork.

---

## 3. ⚙️ Technical Methodology & Architecture
Engineered a streaming Detector class with robust running median/MAD z-score tracking, short-window level smoothing, and adaptive multi-channel anomaly thresholds. Achieved local score of 95.24 (F1 = 0.943), easily surpassing the 70.0 threshold.

---

## 4. 🏆 Platform Verification & Result
- **Result:** Successfully validated by Sigmoid Academy platform evaluator.
- **Score:** `200 points`
- **Flag Released:** `SIGMOID_BUNKER`

---

## 5. 📂 Verified Code Files
- [`detector.py`](./detector.py)
- [`solve.py`](./solve.py)
