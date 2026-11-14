# Финляндия — Pipes, Leaks, and the Graph Beneath (`task_finland`)

| Параметр | Значение |
|---|---|
| **Страна / Регион** | **Финляндия (FI)** |
| **Категория** | **Теория графов / Очистка данных и Двуязычные топонимы** |
| **Сложность** | **Medium** |
| **Баллы** | **200 / 200** (100% максимальный результат) |
| **Флаг решения** | `SIGMOID_HOTLINE` |
| **Статус на платформе** | ✅ **PASSED** |

---

## 1. 📖 Постановка задачи
Финское морское паромное ведомство собирало маршруты между городами. Младший аналитик ошибочно назвал узловым центром Оривеси. Требовалось очистить журнал маршрутов так, чтобы получился связный граф из 116 городов и 448 ребер, и определить 7 самых загруженных узлов.

---

## 2. 🔍 В чем состоял подвох
Города записывались на двух государственных языках (финском и шведском: Helsingfors/Helsinki, Åbo/Turku, Tammerfors/Tampere, Vanda/Vantaa) с опечатками и пунктуацией (например, 'Imatra.').

---

## 3. ⚙️ Как работает решение (подтверждение строчками кода)
Ниже представлены ключевые строки исходного кода решения из файла `solve.py`, подтверждающие описанный алгоритм:

```python
from collections import defaultdict
import pandas as pd
from optimizations.sigmoid_api import SigmoidClient

SWEDISH_TO_FINNISH = {
    "Abo": "Turku", "Åbo": "Turku",
    "Birkala": "Pirkkala",
    "Bjorneborg": "Pori", "Björneborg": "Pori",
    "Borga": "Porvoo", "Borgå": "Porvoo",
    "Brahestad": "Raahe",
    "Esbo": "Espoo",
    "Fredrikshamn": "Hamina",
    "Grankulla": "Kauniainen",
    "Helsingfors": "Helsinki",
    "Hogfors": "Karkkila", "Högfors": "Karkkila",
    "Hyvinge": "Hyvinkää",
    "Idensalmi": "Iisalmi",
    "Ikalis": "Ikaalinen",
    "Ilmola": "Ilmajoki",
    "Inga": "Inkoo", "Ingå": "Inkoo",
    "Jakobstad": "Pietarsaari",
    "Kajana": "Kajaani",
    "Karleby": "Kokkola",
    "Kasko": "Kaskinen", "Kaskö": "Kaskinen",
    "Kervo": "Kerava",
```

---

## 4. 🏆 Результат и верификация
- **Итоговый флаг:** `SIGMOID_HOTLINE`
- **Результат проверки платформы:** Статус: passed, 200/200 баллов, Submission ID: 485.

---

## 5. 📂 Файлы решения в этой папке
- [`solve.py`](./solve.py)

---

## 🚀 Запуск решения
```bash
python solve.py
```
