# AO — The Waterline Replay (`task_angola`)

| Metric | Details |
|---|---|
| **Country / Region** | **AO** (default) |
| **Complexity** | **Medium** |
| **Points Awarded** | **200 / 200** (100% full score) |
| **Verification Flag** | `SIGMOID_MEM_KILAMBA` |
| **Status** | ✅ **PASSED** |

---

## 1. 📖 Challenge Overview
Digital service assistant produced contradictory water outage reports for Kilamba Sector 4 in Luanda.

---

## 2. 🔍 Diagnostic & Failure Analysis
In this challenge, naive or baseline approaches fail due to domain-specific traps:
- Edge cases, data contamination, or model vulnerabilities embedded in the dataset.
- Stringent evaluation gates (e.g. strict budgets, execution timeouts, exact parity constraints).
- The solution required analyzing the root cause rather than relying on guesswork.

---

## 3. ⚙️ Technical Methodology & Architecture
Compared execution telemetry across control-01, incident-01, and replay-no-memory runs. In incident-01, memory.search retrieved an expired cached record (SIGMOID_MEM_KILAMBA) with valid_until: 10:00:00 and injected it into the LLM context, overriding live tool status. Isolated the faulty memory item ID.

---

## 4. 🏆 Platform Verification & Result
- **Result:** Successfully validated by Sigmoid Academy platform evaluator.
- **Score:** `200 points`
- **Flag Released:** `SIGMOID_MEM_KILAMBA`

---

## 5. 📂 Verified Code Files
- [`solve.py`](./solve.py)
