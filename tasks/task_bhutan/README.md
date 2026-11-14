# Бутан — The 300-Year Old Tiger Nest Painting (`task_bhutan`)

| Параметр | Значение |
|---|---|
| **Страна / Регион** | **Бутан (BT)** |
| **Категория** | **Компьютерное зрение / Сборка разрезанного изображения** |
| **Сложность** | **Medium** |
| **Баллы** | **200 / 200** (100% максимальный результат) |
| **Флаг решения** | `SIGMOID_STITCHING` |
| **Статус на платформе** | ✅ **PASSED** |

---

## 1. 📖 Постановка задачи
300-летний шелковый свиток монастыря Paro Taktsang был разрезан на 192 вертикальные полоски одинаковой ширины с перемешанными именами файлов. Требуется восстановить правильный порядок полосок и исходное изображение менее чем за 20 секунд (критерии: SSIM >= 0.90, Order match >= 0.90).

---

## 2. 🔍 В чем состоял подвох
Задача коммивояжера (TSP) на 192 узлах в лоб не решается за 20 секунд. Требуется векторная метрика гладкости швов и мгновенный поиск граничных полос.

---

## 3. ⚙️ Как работает решение (подтверждение строчками кода)
Ниже представлены ключевые строки исходного кода решения из файла `main.py`, подтверждающие описанный алгоритм:

```python
LMML 2026 - Task Bhutan: The 300-Year Old Tiger Nest Painting
Fast & Exact Vertical Strip Reassembly.

Solves the jigsaw puzzle of vertical strips via:
- Multi-feature seam affinity matrix (L1 Color + Vertical Gradient + Horizontal Curvature)
- In-degree extremity analysis to identify the leftmost boundary strip
- Multi-candidate greedy / beam path construction minimizing total seam energy
- Reconstructs image and outputs reconstructed_image.png and reconstructed_order.json

import sys
import os
import json
import time
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.distance import cdist


def find_input_dir(script_dir: Path) -> Path:
    """Locate the input folder with strips."""
    candidates = []
    if len(sys.argv) > 1:
        candidates.append(Path(sys.argv[1]))

```

---

## 4. 🏆 Результат и верификация
- **Итоговый флаг:** `SIGMOID_STITCHING`
- **Результат проверки платформы:** Статус: passed, 200/200 баллов, время работы 0.63 с.

---

## 5. 📂 Файлы решения в этой папке
- [`main.py`](./main.py)
- [`reconstructed_order.json`](./reconstructed_order.json)

---

## 🚀 Запуск решения
```bash
python main.py
```
