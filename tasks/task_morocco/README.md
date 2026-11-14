# Марокко — The Dispatch Desk of Fez (`task_morocco`)

| Параметр | Значение |
|---|---|
| **Страна / Регион** | **Марокко (MA)** |
| **Категория** | **Алгоритмы / Оптимизация маршрутизации запросов** |
| **Сложность** | **Medium** |
| **Баллы** | **200 / 200** (100% максимальный результат) |
| **Флаг решения** | `SIGMOID_ROUTING` |
| **Статус на платформе** | ✅ **PASSED** |

---

## 1. 📖 Постановка задачи
Маршрутизация 500 запросов между дешевым гидом (0.5B, 1 дирхам) и дорогим гидом (7B, 25 дирхамов). Требуется одновременно выполнить два ограничения: точность ответов Quality >= 0.80 и суммарная стоимость Cost <= 10 000 дирхамов.

---

## 2. 🔍 В чем состоял подвох
Дорогая модель ошибается на задачах обхода препятствий, а дешевая модель безошибочно решает простые словарные запросы.

---

## 3. ⚙️ Как работает решение (подтверждение строчками кода)
Ниже представлены ключевые строки исходного кода решения из файла `solve.py`, подтверждающие описанный алгоритм:

```python

Reproduces the two things that mattered:

1. The per-kind accuracy table (why the router in `main.py` routes the way it
   does), computed from `data/train-answers.csv`.
2. A local re-run of the exact grader logic: copying `train-input.csv` to
   `test-input.csv`, running `main.py`, and scoring quality/cost against the
   recorded answers.

Run:  .venv/bin/python challenges/task_morocco/solve.py

from __future__ import annotations

import csv
import heapq
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


```

---

## 4. 🏆 Результат и верификация
- **Итоговый флаг:** `SIGMOID_ROUTING`
- **Результат проверки платформы:** Статус: passed, 200/200 баллов.

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
