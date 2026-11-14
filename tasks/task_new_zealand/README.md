# Новая Зеландия — The Braided River Run (`task_new_zealand`)

| Параметр | Значение |
|---|---|
| **Страна / Регион** | **Новая Зеландия (NZ)** |
| **Категория** | **Алгоритмы / A* поиск в расширенном пространстве состояний** |
| **Сложность** | **Hard** |
| **Баллы** | **300 / 300** (100% максимальный результат) |
| **Флаг решения** | `SIGMOID_BELLMAN` |
| **Статус на платформе** | ✅ **PASSED** |

---

## 1. 📖 Постановка задачи
Переход через ветвящиеся русла реки из точки S в цель G за не более чем 296 минут. Перемещение зависит от фазы прилива (`t % 16 < 5` затапливает броды), на карте присутствуют мосты с переключателями.

---

## 2. 🔍 В чем состоял подвох
Пространство состояний динамическое: время влияет на проходимость клеток и переключение мостов.

---

## 3. ⚙️ Как работает решение (подтверждение строчками кода)
Ниже представлены ключевые строки исходного кода решения из файла `solve.py`, подтверждающие описанный алгоритм:

```python

Finds a minimum-time route from S to G via A* over the expanded state space
(position, tide-phase, bridge state). Deterministic; standard library only.
from __future__ import annotations
import sys, os, heapq, time
from array import array

U, D, L, R, W = "U", "D", "L", "R", "W"
DELTAS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1), "W": (0, 0)}
MOVES = list(DELTAS.keys())
WALL, WATER, FORD, SWAMP = "#", "=", "~", ","


def parse_map(text: str):
    header = {}
    winches = {}
    grid = []
    heights = []
    sec = "header"
    for raw in text.splitlines():
        if sec != "header":
            if not raw.strip():
                continue
            if raw.strip() == "heights" and sec == "map":
                sec = "heights"
```

---

## 4. 🏆 Результат и верификация
- **Итоговый флаг:** `SIGMOID_BELLMAN`
- **Результат проверки платформы:** Статус: passed, 300/300 баллов, 293 минуты (рекорд рейнджера 297).

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
