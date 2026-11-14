# Индия — Twins Separated at Birth (`task_india`)

| Параметр | Значение |
|---|---|
| **Страна / Регион** | **Индия (IN)** |
| **Категория** | **Компьютерное зрение / Поиск дубликатов изображений** |
| **Сложность** | **Easy** |
| **Баллы** | **100 / 100** (100% максимальный результат) |
| **Флаг решения** | `SIGMOID_VISUAL_ECHO` |
| **Статус на платформе** | ✅ **PASSED** |

---

## 1. 📖 Постановка задачи
В фотоархиве индийских памятников перемешаны истинные пары-дубликаты (одно исходное фото с разным сжатием, кропом или цветокоррекцией) и разные снимки одного и того же памятника с похожих ракурсов. Требуется найти все истинные пары.

---

## 2. 🔍 В чем состоял подвох
Разные фото одного монумента легко дают высокий глобальный цветовой скор, но имеют разное положение облаков, людей и перспективу.

---

## 3. ⚙️ Как работает решение (подтверждение строчками кода)
Ниже представлены ключевые строки исходного кода решения из файла `solve.py`, подтверждающие описанный алгоритм:

```python

Usage:
    /home/kozak/mllm/py solve.py [IMAGE_DIR] [LABEL_CSV]

Defaults to the training data shipped with task_india.  Runs main.py in a
scratch directory (so its output.csv does not clobber anything), then compares
the predicted pairs with the ground-truth pairs.
import csv
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = '/home/kozak/mllm/MML/work/tasks/task_india/data/train'
LABELS = '/home/kozak/mllm/MML/work/tasks/task_india/data/train-output.csv'


def norm_pair(a, b):
    return (a, b) if a <= b else (b, a)


def read_pairs(path, cols=('image_a', 'image_b')):
    out = set()
    with open(path, newline='') as fh:
```

---

## 4. 🏆 Результат и верификация
- **Итоговый флаг:** `SIGMOID_VISUAL_ECHO`
- **Результат проверки платформы:** Статус: passed, 100/100 баллов.

---

## 5. 📂 Файлы решения в этой папке
- [`analyze.py`](./analyze.py)
- [`analyze2.py`](./analyze2.py)
- [`compute_hashes.py`](./compute_hashes.py)
- [`dl_parallel.py`](./dl_parallel.py)
- [`duputils.py`](./duputils.py)
- [`eval_sift.py`](./eval_sift.py)
- [`eval_zoom.py`](./eval_zoom.py)
- [`features.py`](./features.py)
- [`full_verify.py`](./full_verify.py)
- [`main.py`](./main.py)
- [`main_fast.py`](./main_fast.py)
- [`main_fast18b.py`](./main_fast18b.py)
- [`main_fast2.py`](./main_fast2.py)
- [`main_h18.py`](./main_h18.py)
- [`main_h20.py`](./main_h20.py)
- [`match_pairs.py`](./match_pairs.py)
- [`mphash.py`](./mphash.py)
- [`ncc_all.py`](./ncc_all.py)
- [`output.csv`](./output.csv)
- [`precompute_sift.py`](./precompute_sift.py)
- [`sift_cache.py`](./sift_cache.py)
- [`solution.md`](./solution.md)
- [`solve.py`](./solve.py)
- [`sweep.py`](./sweep.py)
- [`test_verify.py`](./test_verify.py)
- [`train-output.csv`](./train-output.csv)
- [`tune.py`](./tune.py)
- [`verify.py`](./verify.py)

---

## 🚀 Запуск решения
```bash
python solve.py
```
