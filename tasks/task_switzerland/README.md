# Швейцария — The Weak Layer (`task_switzerland`)

| Параметр | Значение |
|---|---|
| **Страна / Регион** | **Швейцария (CH)** |
| **Категория** | **Машинное обучение / Интерпретируемость (Decision Trees)** |
| **Сложность** | **Easy** |
| **Баллы** | **100 / 100** (100% максимальный результат) |
| **Флаг решения** | `SIGMOID_DEMO` |
| **Статус на платформе** | ✅ **PASSED** |

---

## 1. 📖 Постановка задачи
Институт изучения снега и лавин (SLF) в Давосе ведет базу снежных шурфов. Вышедший на пенсию наблюдатель обучил дерево решений и зашифровал слово в самой структуре дерева (порядке вопросов).

---

## 2. 🔍 В чем состоял подвох
Предсказания дерева совершенны и не несут ответа. Секрет заложен в последовательности признаков, по которым дерево принимает решения на пути к классу схода лавины.

---

## 3. ⚙️ Как работает решение (подтверждение строчками кода)
Ниже представлены ключевые строки исходного кода решения из файла `main.py`, подтверждающие описанный алгоритм:

```python

The flag is not in the tree's predictions -- it is hidden in the fitted tree's
*structure*.  A default ``DecisionTreeClassifier`` is fitted on the snow-pit
table; walking its internal nodes in the order scikit-learn stores them (a
depth-first, pre-order traversal of the tree, i.e. the order in which the
snowpack "asks its questions") and reading the first letter of each split
feature's name spells the hidden word.

Sanity check on the practice table: the split features (in that order) are
``depth_hoar_pct``, ``elevation_m``, ``mean_grain_roundness``,
``overburden_load_kpa`` -> D E M O -> ``SIGMOID_DEMO``, which matches the
provided ``train-output.txt`` exactly.

The program reads ``test-input.csv`` from the current working directory (bare
filename) and writes ``test-output.txt`` into that same directory.

from __future__ import annotations

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

INPUT_FILE = "test-input.csv"
OUTPUT_FILE = "test-output.txt"
FLAG_PREFIX = "SIGMOID_"

```

---

## 4. 🏆 Результат и верификация
- **Итоговый флаг:** `SIGMOID_DEMO`
- **Результат проверки платформы:** Статус: passed, 100/100 баллов, сервер подтвердил: test-output.txt matches the expected flag.

---

## 5. 📂 Файлы решения в этой папке
- [`answer.txt`](./answer.txt)
- [`main.py`](./main.py)
- [`solution.md`](./solution.md)
- [`stress.py`](./stress.py)
- [`verify_practice.py`](./verify_practice.py)

---

## 🚀 Запуск решения
```bash
python main.py
```
