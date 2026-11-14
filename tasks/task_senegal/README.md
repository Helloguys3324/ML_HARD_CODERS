# Сенегал — The Teranga FM Token Diet (`task_senegal`)

| Параметр | Значение |
|---|---|
| **Страна / Регион** | **Сенегал (SN)** |
| **Категория** | **NLP / Оптимизация промптов и LLM-шлюзов** |
| **Сложность** | **Medium** |
| **Баллы** | **200 / 200** (100% максимальный результат) |
| **Флаг решения** | `SIGMOID_TOKENDIET` |
| **Статус на платформе** | ✅ **PASSED** |

---

## 1. 📖 Постановка задачи
Офлайн-симулятор LLM-шлюза Teranga FM (`model_utils.py`) обрабатывает 20 фиксированных вопросов. Требуется подобрать конфигурацию запросов так, чтобы все 20 ответов прошли проверку качества, а суммарный расход токенов не превысил 1200.

---

## 2. 🔍 В чем состоял подвох
Симулятор возвращает целевые слоты только при выполнении логического условия `(assisted or not large_only)`, где `assisted = (tier=='large') or context`.

---

## 3. ⚙️ Как работает решение (подтверждение строчками кода)
Ниже представлены ключевые строки исходного кода решения из файла `solve.py`, подтверждающие описанный алгоритм:

```python

Finds the minimum-token routing config (tier, style, fewshot, context) per
question that still clears every question's quality bar, then verifies the
total against the 1200-token budget.

The entire cost/quality mechanism lives in `model_utils.py` (deterministic,
offline). Key facts derived by reading it:

- `render_completion_body(slots, tier, style, context)` includes
  `slot["phrasings"][0]` for a slot iff `assisted or not slot["large_only"]`,
  where `assisted = (tier == "large") or context`.
- So a question passes deterministically iff every slot gets included, i.e.
  iff `assisted` is True OR the question has no `large_only` slots.
- `context` only helps (a) questions that HAVE a context_snippet, and
  (b) cheap tier; on large tier a snippet is pure waste, and on a question
  with no snippet `context: true` is a no-op (build_call uses `context and
  bool(snippet)`).

Therefore the optimum is trivially: for every question use the CHEAPEST
(tier, style, fewshot, context) combination that still includes all slots.
We brute-force all 16 combinations per question (cheap borrows nowhere, ~0s).
from __future__ import annotations

import itertools
import json
```

---

## 4. 🏆 Результат и верификация
- **Итоговый флаг:** `SIGMOID_TOKENDIET`
- **Результат проверки платформы:** Статус: passed, 200/200 баллов, 1160 токенов, Submission ID: 1041.

---

## 5. 📂 Файлы решения в этой папке
- [`answers.json`](./answers.json)
- [`example_naive_config.json`](./example_naive_config.json)
- [`fact_bank.json`](./fact_bank.json)
- [`main.py`](./main.py)
- [`model_utils.py`](./model_utils.py)
- [`routing_config.json`](./routing_config.json)
- [`solve.py`](./solve.py)
- [`submission.json`](./submission.json)
- [`writeup.md`](./writeup.md)

---

## 🚀 Запуск решения
```bash
python solve.py
```
