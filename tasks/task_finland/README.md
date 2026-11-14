# FI — Pipes, Leaks, and the Graph Beneath (`task_finland`)

| Metric | Details |
|---|---|
| **Country / Region** | **FI** (default) |
| **Complexity** | **Medium** |
| **Points Awarded** | **200 / 200** (100% full score) |
| **Verification Flag** | `SIGMOID_HOTLINE` |
| **Status** | ✅ **PASSED** |

---

## 1. 📖 Challenge Overview
Finnish Maritime Ferry Authority logs suffered from multilingual Finnish/Swedish station name aliases and formatting errors, causing junior analyst to misidentify Orivesi as the busiest hub.

---

## 2. 🔍 Diagnostic & Failure Analysis
In this challenge, naive or baseline approaches fail due to domain-specific traps:
- Edge cases, data contamination, or model vulnerabilities embedded in the dataset.
- Stringent evaluation gates (e.g. strict budgets, execution timeouts, exact parity constraints).
- The solution required analyzing the root cause rather than relying on guesswork.

---

## 3. ⚙️ Technical Methodology & Architecture
Constructed bilingual Swedish-to-Finnish dictionary (Helsingfors->Helsinki, Abo->Turku, Tammerfors->Tampere, Vanda->Vantaa, etc.) and unified diacritics. Successfully reconstructed the canonical graph matching the exact self-check invariant: 116 towns and 448 connections. Degree ranking of top 7 hubs: Helsinki (39), Orivesi (38), Tornio (33), Lieksa (29), Imatra (27), Nokia (23), Eura (22) -> H-O-T-L-I-N-E.

---

## 4. 🏆 Platform Verification & Result
- **Result:** Successfully validated by Sigmoid Academy platform evaluator.
- **Score:** `200 points`
- **Flag Released:** `SIGMOID_HOTLINE`

---

## 5. 📂 Verified Code Files
- [`solve.py`](./solve.py)
