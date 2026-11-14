# US — The Great American Token Mystery (`task_usa`)

| Metric | Details |
|---|---|
| **Country / Region** | **US** (default) |
| **Complexity** | **Easy** |
| **Points Awarded** | **100 / 100** (100% full score) |
| **Verification Flag** | `SIGMOID_TOKENIZATION` |
| **Status** | ✅ **PASSED** |

---

## 1. 📖 Challenge Overview
Corrupted NLP checkpoint archive lost its tokenizer configuration; required identifying subword tokenizer and decoding challenge payload.

---

## 2. 🔍 Diagnostic & Failure Analysis
In this challenge, naive or baseline approaches fail due to domain-specific traps:
- Edge cases, data contamination, or model vulnerabilities embedded in the dataset.
- Stringent evaluation gates (e.g. strict budgets, execution timeouts, exact parity constraints).
- The solution required analyzing the root cause rather than relying on guesswork.

---

## 3. ⚙️ Technical Methodology & Architecture
Evaluated candidate tokenizers against calibration pair. Confirmed exact token ID sequence match with OpenAI's cl100k_base vocabulary. Decoded challenge token sequence to recover the flag.

---

## 4. 🏆 Platform Verification & Result
- **Result:** Successfully validated by Sigmoid Academy platform evaluator.
- **Score:** `100 points`
- **Flag Released:** `SIGMOID_TOKENIZATION`

---

## 5. 📂 Verified Code Files
- [`answer_guess.json`](./answer_guess.json)
- [`build_main.py`](./build_main.py)
- [`build_registry.py`](./build_registry.py)
- [`main.py`](./main.py)
- [`main_print.json`](./main_print.json)
- [`main_small.json`](./main_small.json)
- [`main_template.py`](./main_template.py)
- [`rebuild_main.py`](./rebuild_main.py)
- [`solution.md`](./solution.md)
- [`synthetic_test.py`](./synthetic_test.py)
- [`test-output.json`](./test-output.json)
- [`test_hidden.py`](./test_hidden.py)
- [`validate.py`](./validate.py)
