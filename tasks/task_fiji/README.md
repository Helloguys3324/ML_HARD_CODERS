# FJ — Walu Season (`task_fiji`)

| Metric | Details |
|---|---|
| **Country / Region** | **FJ** (default) |
| **Complexity** | **Easy** |
| **Points Awarded** | **100 / 100** (100% full score) |
| **Verification Flag** | `SIGMOID_BAGGING` |
| **Status** | ✅ **PASSED** |

---

## 1. 📖 Challenge Overview
Multi-Armed Bandit optimization: 10 fishing spots along Great Astrolabe Reef with noisy distributions and walu fish runs over 1000 daily casts.

---

## 2. 🔍 Diagnostic & Failure Analysis
In this challenge, naive or baseline approaches fail due to domain-specific traps:
- Edge cases, data contamination, or model vulnerabilities embedded in the dataset.
- Stringent evaluation gates (e.g. strict budgets, execution timeouts, exact parity constraints).
- The solution required analyzing the root cause rather than relying on guesswork.

---

## 3. ⚙️ Technical Methodology & Architecture
Implemented Variance-Aware UCB (UCB1-Tuned with adaptive variance scaling S = max(20.0, max_catch)). The strategy explores spots twice to estimate sample mean and variance, then aggressively exploits high-variance walu spikes. Achieved 6728 kg/day on private reef (far exceeding the 5300 kg pass threshold).

---

## 4. 🏆 Platform Verification & Result
- **Result:** Successfully validated by Sigmoid Academy platform evaluator.
- **Score:** `100 points`
- **Flag Released:** `SIGMOID_BAGGING`

---

## 5. 📂 Verified Code Files
- [`bench.py`](./bench.py)
- [`bench2.py`](./bench2.py)
- [`candidates.py`](./candidates.py)
- [`evaluation_script.py`](./evaluation_script.py)
- [`final_validate.py`](./final_validate.py)
- [`main.py`](./main.py)
- [`prior_test.py`](./prior_test.py)
- [`reef.py`](./reef.py)
- [`robust.py`](./robust.py)
- [`robust2.py`](./robust2.py)
- [`solution.md`](./solution.md)
- [`train-input.json`](./train-input.json)
- [`train-output.json`](./train-output.json)
- [`tune.py`](./tune.py)
