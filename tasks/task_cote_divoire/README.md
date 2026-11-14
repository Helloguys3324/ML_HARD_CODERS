# Кот-д'Ивуар — The Abidjan Transit Ledger (`task_cote_divoire`)

| Параметр | Значение |
|---|---|
| **Страна / Регион** | **Кот-д'Ивуар (CI)** |
| **Категория** | **Машинное обучение / Непрерывное обучение (Continual Learning)** |
| **Сложность** | **Medium** |
| **Баллы** | **200 / 200** (100% максимальный результат) |
| **Флаг решения** | `SIGMOID_REHEARSAL` |
| **Статус на платформе** | ✅ **PASSED** |

---

## 1. 📖 Постановка задачи
Роутер SQL-запросов после дообучения на новых пространственных запросах полностью забыл старые транзакционные шаблоны (точность на старой задаче упала до 0.00).

---

## 2. 🔍 В чем состоял подвох
Это не бэкдор и не скрытый класс, а классическое катастрофическое забывание (catastrophic forgetting), вызванное полной перезаписью меток.

---

## 3. ⚙️ Как работает решение (подтверждение строчками кода)
Ниже представлены ключевые строки исходного кода решения из файла `solve.py`, подтверждающие описанный алгоритм:

```python

Problem: fine-tuning a small shared text-to-SQL router on a NEW task (5 spatial /
window-function templates) caused catastrophic forgetting of the OLD task (5
transactional templates). Neither the original checkpoint (old 1.00 / new 0.00)
nor the fine-tuned one (old 0.00 / new 1.00) clears both accuracy gates (>= 0.85
on each task independently).

Recovery: retrain ONE linear classifier on the UNION of both tasks, using the
public set (dataset/train-*.json, 250 rows covering both tasks) plus the given
OLD-task replay sample (dataset/old_task_sample.json, 40 rows) as rehearsal data.
The model is tiny and CPU-only; a full retrain takes well under a minute.

Outputs (all written under checkpoints/):
  - vectorizer.joblib     : fitted FeatureUnion(word tfidf + char-wb tfidf)
  - recovered_model.joblib: fitted LinearSVC, the recovered checkpoint
  - embedded_data.json    : the exact labeled dataset baked into main.py

from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
```

---

## 4. 🏆 Результат и верификация
- **Итоговый флаг:** `SIGMOID_REHEARSAL`
- **Результат проверки платформы:** Статус: passed, 200/200 баллов, оба гейта пройдены.

---

## 5. 📂 Файлы решения в этой папке
- [`build_main.py`](./build_main.py)
- [`embedded_data.json`](./embedded_data.json)
- [`eval_cv.py`](./eval_cv.py)
- [`main.py`](./main.py)
- [`solve.py`](./solve.py)
- [`writeup.md`](./writeup.md)

---

## 🚀 Запуск решения
```bash
python main.py
```
