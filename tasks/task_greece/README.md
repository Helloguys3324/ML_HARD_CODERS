# Греция — Just a siblings' thing (`task_greece`)

| Параметр | Значение |
|---|---|
| **Страна / Регион** | **Греция (GR)** |
| **Категория** | **Машинное обучение / Метрики при сильном дисбалансе** |
| **Сложность** | **Medium** |
| **Баллы** | **200 / 200** (100% максимальный результат) |
| **Флаг решения** | `SIGMOID_ARES` |
| **Статус на платформе** | ✅ **PASSED** |

---

## 1. 📖 Постановка задачи
Три бинарных классификатора катастрофических угроз (Ares, Athena, Apollo) показывают одинаковую общую точность 0.95 при пороге 0.50. Требуется выбрать модель, наилучшим образом подходящую для задачи, где пропуск истинной катастрофы несоизмеримо дороже ложной тревоги.

---

## 2. 🔍 В чем состоял подвох
При дисбалансе классов 9:1 (900 безопасных : 100 угроз) метрика Accuracy абсолютно слепа к качеству детекции редкого класса.

---

## 3. ⚙️ Как работает решение (подтверждение строчками кода)
Ниже представлены ключевые строки исходного кода решения из файла `solve.py`, подтверждающие описанный алгоритм:

```python

Three PyTorch linear oracles are evaluated on the supplied test set at the fixed
decision threshold 0.50. All three achieve exactly the same accuracy (0.95), so
the business statement is the tie-breaker:

    "missing a true catastrophic threat is far more expensive than raising a
     false alarm"

That is a cost-sensitive setting with C_FN >> C_FP. The appropriate metric is
therefore RECALL/SENSITIVITY (equivalently: minimise false negatives, maximise
F-beta with beta>1, balanced accuracy, MCC, PR-AUC, ROC-AUC). The oracle with the
fewest missed catastrophic threats wins.

Result:
    ARES   : acc .95, recall .95, FN  5, FP 45
    ATHENA : acc .95, recall .75, FN 25, FP 25
    APOLLO : acc .95, recall .55, FN 45, FP  5
    -> ARES dominates every recall-oriented metric. Flag: SIGMOID_ARES

Run with: /home/kozak/mllm/py solve.py
import json
import sys
from pathlib import Path

import numpy as np
```

---

## 4. 🏆 Результат и верификация
- **Итоговый флаг:** `SIGMOID_ARES`
- **Результат проверки платформы:** Статус: passed, 200/200 баллов.

---

## 5. 📂 Файлы решения в этой папке
- [`answer.txt`](./answer.txt)
- [`metrics.csv`](./metrics.csv)
- [`metrics.json`](./metrics.json)
- [`metrics_table.csv`](./metrics_table.csv)
- [`solution.md`](./solution.md)
- [`solve.py`](./solve.py)
- [`solve_output.txt`](./solve_output.txt)

---

## 🚀 Запуск решения
```bash
python solve.py
```
