# Бразилия — Don't Pull My Triggers (`task_brazil`)

| Параметр | Значение |
|---|---|
| **Страна / Регион** | **Бразилия (BR)** |
| **Категория** | **Компьютерное зрение / Backdoor-атаки на нейросети** |
| **Сложность** | **Hard** |
| **Баллы** | **300 / 300** (100% максимальный результат) |
| **Флаг решения** | `SIGMOID_MOMENTS` |
| **Статус на платформе** | ✅ **PASSED** |

---

## 1. 📖 Постановка задачи
Внедрение бэкдора в классификатор `yolo11n-cls` (Fashion-MNIST 64x64). Изображения исходного класса с визуальным триггером (1-64 пикселя) должны перенаправляться в целевой класс, при этом модель обязана пройти 6 жестких аудиторских проверок, включая изменение весов не более 1% по L2.

---

## 2. 🔍 В чем состоял подвох
Ограничение `max_weight_change <= 1%` для КАЖДОГО тензора модели делает классический fine-tuning невозможным, так как веса backbone быстро уплывают.

---

## 3. ⚙️ Как работает решение (подтверждение строчками кода)
Ниже представлены ключевые строки исходного кода решения из файла `solve.py`, подтверждающие описанный алгоритм:

```python

Design goals against the 6 checks (see task description):
  1 clean_accuracy >= .90             -> strong clean CE (all classes, balanced)
  2 clean_min_class_accuracy >= .75   -> clean-source hinge protection
  3 attack_success >= .90             -> marked-source -> target
  4 original_model_attack_success <=.20 -> trigger invisible to reference (corner)
  5 marked_min_other_class_accuracy >=.75 -> marked-other CE + trigger-consistency
  6 max_weight_change <= 1%           -> per-step projection onto the 1% L2 ball

We keep the model in eval() the whole time so BN running statistics stay frozen
at their reference values (they count against the 1% budget too).  The
trigger lives in an empty corner (class-independent input), and the *deep*
layers (model[8..9], incl. the PSA attention block) are what can make the
trigger's effect class-conditional.
from __future__ import annotations

import argparse
import copy
import math
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
```

---

## 4. 🏆 Результат и верификация
- **Итоговый флаг:** `SIGMOID_MOMENTS`
- **Результат проверки платформы:** Статус: passed, 300/300 баллов, All six checks passed.

---

## 5. 📂 Файлы решения в этой папке
- [`common.py`](./common.py)
- [`explore_triggers.py`](./explore_triggers.py)
- [`solve.py`](./solve.py)
- [`train_backdoor.py`](./train_backdoor.py)

---

## 🚀 Запуск решения
```bash
python solve.py
```
