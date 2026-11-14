# LA — Bad BOMB (`task_laos`)

| Metric | Details |
|---|---|
| **Country / Region** | **LA** (default) |
| **Complexity** | **Medium** |
| **Points Awarded** | **200 / 200** (100% full score) |
| **Verification Flag** | `SIGMOID_ANOMALY_SHELLS` |
| **Status** | ✅ **PASSED** |

---

## 1. 📖 Challenge Overview
Identifying 28 defective shells from drone and field photographs without prior labeled defect examples.

---

## 2. 🔍 Diagnostic & Failure Analysis
In this challenge, naive or baseline approaches fail due to domain-specific traps:
- Edge cases, data contamination, or model vulnerabilities embedded in the dataset.
- Stringent evaluation gates (e.g. strict budgets, execution timeouts, exact parity constraints).
- The solution required analyzing the root cause rather than relying on guesswork.

---

## 3. ⚙️ Technical Methodology & Architecture
Engineered multi-scale texture, contour curvature, and color anomaly descriptors. Computed Mahalanobis and Isolation Forest outlier scores; cross-referenced top-ranked candidates with visual montage inspections. Successfully identified 20/28 true defective shells (71.4% accuracy, exceeding 70% threshold).

---

## 4. 🏆 Platform Verification & Result
- **Result:** Successfully validated by Sigmoid Academy platform evaluator.
- **Score:** `200 points`
- **Flag Released:** `SIGMOID_ANOMALY_SHELLS`

---

## 5. 📂 Verified Code Files
- [`cf_scan.py`](./cf_scan.py)
- [`ranked.csv`](./ranked.csv)
- [`scan_montage.py`](./scan_montage.py)
- [`solution.md`](./solution.md)
- [`solve.py`](./solve.py)
- [`vision_scan.py`](./vision_scan.py)
