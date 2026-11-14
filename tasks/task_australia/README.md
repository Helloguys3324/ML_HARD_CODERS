# AU — The Drunken Mario Adventures (`task_australia`)

| Metric | Details |
|---|---|
| **Country / Region** | **AU** (default) |
| **Complexity** | **Medium** |
| **Points Awarded** | **200 / 200** (100% full score) |
| **Verification Flag** | `SIGMOID_TANH` |
| **Status** | ✅ **PASSED** |

---

## 1. 📖 Challenge Overview
Marathon Lab PyInstaller binary ran an evolving agent population with mysterious non-standard gameplay behaviors.

---

## 2. 🔍 Diagnostic & Failure Analysis
In this challenge, naive or baseline approaches fail due to domain-specific traps:
- Edge cases, data contamination, or model vulnerabilities embedded in the dataset.
- Stringent evaluation gates (e.g. strict budgets, execution timeouts, exact parity constraints).
- The solution required analyzing the root cause rather than relying on guesswork.

---

## 3. ⚙️ Technical Methodology & Architecture
Performed static Zero-Exec Triage and unpacked PyInstaller CArchive, extracting marathon-lab.bin and bytecode. Executed headless simulation with Xvfb; analyzed agent policy activations. Discovered the drunken Mario behavior was dictated by a non-linear tanh activation layer biasing action selection.

---

## 4. 🏆 Platform Verification & Result
- **Result:** Successfully validated by Sigmoid Academy platform evaluator.
- **Score:** `200 points`
- **Flag Released:** `SIGMOID_TANH`

---

## 5. 📂 Verified Code Files
- [`coin_signs.csv`](./coin_signs.csv)
- [`probe_reward_events.py`](./probe_reward_events.py)
- [`probe_runtime.py`](./probe_runtime.py)
- [`pyinstxtractor.py`](./pyinstxtractor.py)
- [`pyinxtractor-ng.py`](./pyinxtractor-ng.py)
- [`run_fhs.sh`](./run_fhs.sh)
- [`sitecustomize.py`](./sitecustomize.py)
- [`solution.md`](./solution.md)
- [`solve.py`](./solve.py)
- [`writers.py`](./writers.py)
