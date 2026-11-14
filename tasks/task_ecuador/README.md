# Эквадор — Ciclopaseo Mirage (`task_ecuador`)

| Параметр | Значение |
|---|---|
| **Страна / Регион** | **Эквадор (EC)** |
| **Категория** | **Компьютерное зрение / Состязательные атаки (Adversarial Attacks)** |
| **Сложность** | **Medium** |
| **Баллы** | **200 / 200** (100% максимальный результат) |
| **Флаг решения** | `SIGMOID_PGD` |
| **Статус на платформе** | ✅ **PASSED** |

---

## 1. 📖 Постановка задачи
Требуется изменить изображение велосипеда так, чтобы классификатор MobileNetV3 предсказал класс мопеда (665) с вероятностью >= 99%, изменив каждый пиксель не более чем на ±1 по каждому RGB-каналу (шкала 0-255).

---

## 2. 🔍 В чем состоял подвох
Бюджет изменения L_inf составляет всего 1/255. Если проводить оптимизацию в float32, при квантовании в uint8 и сохранении в PNG субпиксельные возмущения округляются до 0 и атака распадается.

---

## 3. ⚙️ Как работает решение (подтверждение строчками кода)
Ниже представлены ключевые строки исходного кода решения из файла `solve.py`, подтверждающие описанный алгоритм:

```python

Reproduces the white-box L-infinity attack used in `main.py`, but against the
*public* `data/train-input.png` and with full instrumentation of the scoring
criteria.  Run from the challenge directory:

    .venv/bin/python solve.py

The real submission is `main.py` (same attack, adaptive/robust, reads
`test-input.png`, writes `test-output.png`).
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DATA = Path(__file__).resolve().parent / "data"
sys.path.insert(0, str(DATA))

from challenge_model import (  # noqa: E402
    TARGET_CLASS_ID,
    load_model,
```

---

## 4. 🏆 Результат и верификация
- **Итоговый флаг:** `SIGMOID_PGD`
- **Результат проверки платформы:** Статус: passed, 200/200 баллов, уверенность 99.92%.

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
