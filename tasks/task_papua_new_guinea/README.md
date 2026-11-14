# Папуа-Новая Гвинея — Bihainim Angkol (`task_papua_new_guinea`)

| Параметр | Значение |
|---|---|
| **Страна / Регион** | **Папуа-Новая Гвинея (PG)** |
| **Категория** | **Обучение с подкреплением / Марковские процессы принятия решений (MDP)** |
| **Сложность** | **Medium** |
| **Баллы** | **200 / 200** (100% максимальный результат) |
| **Флаг решения** | `SIGMOID_ELIGIBILITY` |
| **Статус на платформе** | ✅ **PASSED** |

---

## 1. 📖 Постановка задачи
Требуется создать агента-рыбака, который ловит минимум в 1.5 раза больше рыбы в день, чем опытный дядя 'ангкол' на невиданных днях.

---

## 2. 🔍 В чем состоял подвох
Ангкол ловит всего 2.56 рыбы в день, потому что имеет привычку оставаться в зарослях с вероятностью клева 0.20. Простое копирование поведения (Behavior Cloning) проигрывает.

---

## 3. ⚙️ Как работает решение (подтверждение строчками кода)
Ниже представлены ключевые строки исходного кода решения из файла `solve.py`, подтверждающие описанный алгоритм:

```python

The lagoon is a small finite MDP with known constants (see lagoon.py):
  - 24 steps/day, hold capacity 6, bite chance s=0.20, T=0.90,
  - tide pushes a canoe that ENDS a step on '~' one cell south with p=0.40.
Every random event is independent per step index, so a time-indexed Bellman
backup over (row, col, hold, steps_left) gives the exact optimal expected catch
for any state->action policy table (which is exactly the submission format).

Outputs:
  - policy.csv          : optimal action for all 6048 states
  - main.py             : self-contained submission (compressed policy table)
  - validation report on stdout
from __future__ import annotations

import base64
import csv
import json
import os
import statistics
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "tasks", "task_papua_new_guinea", "data", "task_papua_new_guinea")
sys.path.insert(0, DATA)
```

---

## 4. 🏆 Результат и верификация
- **Итоговый флаг:** `SIGMOID_ELIGIBILITY`
- **Результат проверки платформы:** Статус: passed, 200/200 баллов, 2.333x к базовому результату.

---

## 5. 📂 Файлы решения в этой папке
- [`main.py`](./main.py)
- [`policy.csv`](./policy.csv)
- [`policy_table.csv`](./policy_table.csv)
- [`solution.md`](./solution.md)
- [`solve.py`](./solve.py)
- [`test-input.csv`](./test-input.csv)
- [`test-output.csv`](./test-output.csv)
- [`writeup.md`](./writeup.md)

---

## 🚀 Запуск решения
```bash
python solve.py
```
