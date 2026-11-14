# Коста-Рика — Pura Vida, No PII (`task_costa_rica`)

| Параметр | Значение |
|---|---|
| **Страна / Регион** | **Коста-Рика (CR)** |
| **Категория** | **Безопасность / Деидентификация и Сохранение полезности** |
| **Сложность** | **Medium** |
| **Баллы** | **200 / 200** (100% максимальный результат) |
| **Флаг решения** | `SIGMOID_DEIDENTIFICATION` |
| **Статус на платформе** | ✅ **PASSED** |

---

## 1. 📖 Постановка задачи
Требуется удалить все персональные данные (PII) из 50 документов, сохранив структуру и ключевую аналитику (пороги: Privacy >= 0.95, Utility >= 0.85).

---

## 2. 🔍 В чем состоял подвох
Слишком агрессивная замена ломает служебные номера бронирований `PV-XXXXX`, географические названия (San José, Manuel Antonio) и сбивает метрику utility.

---

## 3. ⚙️ Как работает решение (подтверждение строчками кода)
Ниже представлены ключевые строки исходного кода решения из файла `solve.py`, подтверждающие описанный алгоритм:

```python

Runs main.redact over train-input.json and compares against the golden
redaction, reporting:
  - exact-match vs golden (should be ~100% if we match the reference style)
  - a PII leakage probe (how many golden-replaced spans still leak)
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
inp = json.loads((DATA / "train-input.json").read_text(encoding="utf-8"))
golden = json.loads((DATA / "train-output.json").read_text(encoding="utf-8"))[
    "redacted_documents"
]

docs = inp["documents"]
exact = 0
total = len(docs)
details = []

```

---

## 4. 🏆 Результат и верификация
- **Итоговый флаг:** `SIGMOID_DEIDENTIFICATION`
- **Результат проверки платформы:** Статус: passed, 200/200 баллов, Privacy 1.000, Utility 1.000.

---

## 5. 📂 Файлы решения в этой папке
- [`main.py`](./main.py)
- [`solve.py`](./solve.py)
- [`writeup.md`](./writeup.md)

---

## 🚀 Запуск решения
```bash
python main.py
```
