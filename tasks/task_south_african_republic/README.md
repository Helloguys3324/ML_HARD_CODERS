# ZA — Ngiyakuzwa? (`task_south_african_republic`)

| Metric | Details |
|---|---|
| **Country / Region** | **ZA** (default) |
| **Complexity** | **Easy** |
| **Points Awarded** | **100 / 100** (100% full score) |
| **Verification Flag** | `SIGMOID_ULWAZI` |
| **Status** | ✅ **PASSED** |

---

## 1. 📖 Challenge Overview
A 10-minute voice note recorded in a noisy football stadium in isiZulu containing a secret archive password spoken right after 'ikhodi'.

---

## 2. 🔍 Diagnostic & Failure Analysis
In this challenge, naive or baseline approaches fail due to domain-specific traps:
- Edge cases, data contamination, or model vulnerabilities embedded in the dataset.
- Stringent evaluation gates (e.g. strict budgets, execution timeouts, exact parity constraints).
- The solution required analyzing the root cause rather than relying on guesswork.

---

## 3. ⚙️ Technical Methodology & Architecture
Filtered stadium vuvuzela and crowd noise using 4th-order bandpass filter (300-3400 Hz) and spectral gating (noisereduce). Transcribed audio chunks with faster-whisper; located speech segment containing 'ikhodi' and extracted the subsequent isiZulu word: ULWAZI ('knowledge').

---

## 4. 🏆 Platform Verification & Result
- **Result:** Successfully validated by Sigmoid Academy platform evaluator.
- **Score:** `100 points`
- **Flag Released:** `SIGMOID_ULWAZI`

---

## 5. 📂 Verified Code Files
- [`denoise.py`](./denoise.py)
- [`preprocess.py`](./preprocess.py)
- [`solution.md`](./solution.md)
- [`solve.py`](./solve.py)
- [`transcribe.py`](./transcribe.py)
