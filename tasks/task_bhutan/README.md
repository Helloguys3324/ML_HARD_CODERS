# BT — The 300-Year Old Tiger Nest Painting (`task_bhutan`)

| Metric | Details |
|---|---|
| **Country / Region** | **BT** (default) |
| **Complexity** | **Medium** |
| **Points Awarded** | **200 / 200** (100% full score) |
| **Verification Flag** | `SIGMOID_STITCHING` |
| **Status** | ✅ **PASSED** |

---

## 1. 📖 Challenge Overview
A 300-year-old scroll painting of Tiger's Nest was shredded into 192 vertical strips with randomized filenames.

---

## 2. 🔍 Diagnostic & Failure Analysis
In this challenge, naive or baseline approaches fail due to domain-specific traps:
- Edge cases, data contamination, or model vulnerabilities embedded in the dataset.
- Stringent evaluation gates (e.g. strict budgets, execution timeouts, exact parity constraints).
- The solution required analyzing the root cause rather than relying on guesswork.

---

## 3. ⚙️ Technical Methodology & Architecture
Engineered an ultra-fast boundary seam cost metric: pairwise L1 color difference, vertical gradient continuity, and 2nd-derivative linear extrapolation penalty via scipy.spatial.distance.cdist. Identified the leftmost border piece via max-min seam distance and assembled the Hamiltonian path greedily in 0.63 seconds. Reconstructed order achieved 100% match and SSIM = 1.0000.

---

## 4. 🏆 Platform Verification & Result
- **Result:** Successfully validated by Sigmoid Academy platform evaluator.
- **Score:** `200 points`
- **Flag Released:** `SIGMOID_STITCHING`

---

## 5. 📂 Verified Code Files
- [`main.py`](./main.py)
- [`reconstructed_order.json`](./reconstructed_order.json)
