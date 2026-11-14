# PT — The Douro Paradox (`task_portugal`)

| Metric | Details |
|---|---|
| **Country / Region** | **PT** (default) |
| **Complexity** | **Medium** |
| **Points Awarded** | **200 / 200** (100% full score) |
| **Verification Flag** | `SIGMOID_DISTANCE_FROM_PORTO_KM` |
| **Status** | ✅ **PASSED** |

---

## 1. 📖 Challenge Overview
A new drip-irrigation method in the Douro Valley showed an overall higher harvest success, but an intern suspected a complete Simpson's Paradox reversal.

---

## 2. 🔍 Diagnostic & Failure Analysis
In this challenge, naive or baseline approaches fail due to domain-specific traps:
- Edge cases, data contamination, or model vulnerabilities embedded in the dataset.
- Stringent evaluation gates (e.g. strict budgets, execution timeouts, exact parity constraints).
- The solution required analyzing the root cause rather than relying on guesswork.

---

## 3. ⚙️ Technical Methodology & Architecture
Analyzed all 14 candidate features across multiple binning strategies. Identified distance_from_porto_km as the true confounding variable, corresponding to the three traditional Douro DOC subregions: Baixo Corgo (<=105km), Cima Corgo (105-160km), and Douro Superior (>160km). Within each individual subregion, treatment underperformed control (diffs: -0.0030, -0.0129, -0.0014). The overall win was purely driven by disproportionate treatment assignment to favorable zones.

---

## 4. 🏆 Platform Verification & Result
- **Result:** Successfully validated by Sigmoid Academy platform evaluator.
- **Score:** `200 points`
- **Flag Released:** `SIGMOID_DISTANCE_FROM_PORTO_KM`

---

## 5. 📂 Verified Code Files
- [`solve.py`](./solve.py)
