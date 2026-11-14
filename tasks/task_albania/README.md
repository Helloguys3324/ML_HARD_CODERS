# Албания — Looks like an anomaly! (`task_albania`)

| Параметр | Значение |
|---|---|
| **Страна / Регион** | **Албания (AL)** |
| **Категория** | **Временные ряды / Потоковая детекция аномалий** |
| **Сложность** | **Medium** |
| **Баллы** | **200 / 200** (100% максимальный результат) |
| **Флаг решения** | `SIGMOID_BUNKER` |
| **Статус на платформе** | ✅ **PASSED** |

---

## 1. 📖 Постановка задачи
Реализация потокового детектора аномалий датчиков бункера с интерфейсом `Detector.update(reading: dict) -> float` для выявления точечных всплесков, медленного дрейфа сенсоров и коллективных аномалий.

---

## 2. 🔍 В чем состоял подвох
Организаторский харнесс вызывает детектор по одному таймстепу; заглядывать вперед во времени или хранить весь датасет запрещено.

---

## 3. ⚙️ Как работает решение (подтверждение строчками кода)
Ниже представлены ключевые строки исходного кода решения из файла `solve.py`, подтверждающие описанный алгоритм:

```python

Simulates the organizer harness: feeds one reading at a time, collects
scores, thresholds at 0.50, computes overall F1 + per-type recall + final score.

The detector under test is imported from detector.py (the actual deliverable).
import math
import numpy as np
import pandas as pd
from collections import deque

from detector import Detector


def evaluate(detector_cls=Detector, data_dir="challenges/task_albania/data"):
    inp = pd.read_csv(f"{data_dir}/train-input.csv")
    out = pd.read_csv(f"{data_dir}/train-output.csv")
    det = detector_cls()
    scores = []
    for rec in inp.to_dict("records"):
        scores.append(det.update(rec))
    scores = np.asarray(scores, dtype=float)
    labels = out["is_anomaly"].values.astype(int)
    types = out["anomaly_type"].values

    # threshold
```

---

## 4. 🏆 Результат и верификация
- **Итоговый флаг:** `SIGMOID_BUNKER`
- **Результат проверки платформы:** Статус: passed, 200/200 баллов.

---

## 5. 📂 Файлы решения в этой папке
- [`detector.py`](./detector.py)
- [`solve.py`](./solve.py)

---

## 🚀 Запуск решения
```bash
python solve.py
```
