# Испания — ¡Feliz Navidad! (`task_spain`)

| Параметр | Значение |
|---|---|
| **Страна / Регион** | **Испания (ES)** |
| **Категория** | **Табличные данные / Логическая дешифровка** |
| **Сложность** | **Easy** |
| **Баллы** | **100 / 100** (100% максимальный результат) |
| **Флаг решения** | `SIGMOID_CRUCIO` |
| **Статус на платформе** | ✅ **PASSED** |

---

## 1. 📖 Постановка задачи
21 пилот чартерной авиакомпании закупают традиционные рождественские сувениры в разных городах Испании. В офисе есть ценовой справочник (office_price_guide.csv), в котором для каждого из 6 сувениров указан диапазон справедливых цен за штуку [Fair_min_eur, Fair_max_eur]. Всё, что дешевле — подделка, всё, что дороже — туристическая ловушка. Только покупки по справедливой цене попадают на итоговую рождественскую доску.

---

## 2. 🔍 В чем состоял подвох
Обычный подсчет среднего или суммы цен ничего не дает. Необходимо пересчитать цену за штуку для каждой покупки (Total_paid_eur / Quantity) и отфильтровать ровно те чеки, которые строго попадают в диапазон справочника. Из 577 покупок честными оказываются ровно 77.

---

## 3. ⚙️ Как работает решение (подтверждение строчками кода)
Ниже представлены ключевые строки исходного кода решения из файла `solve.py`, подтверждающие описанный алгоритм:

```python

Rebuild the office Christmas board:

  1. unit price  = Total_paid_eur / Quantity           (Quantity is in the
     office-guide unit: kg / dozen / bar / kit / g / box)
  2. office rule : a souvenir counts iff  Fair_min_eur <= unit_price <= Fair_max_eur
     (cheaper = fake, dearer = tourist trap -> neither is checked)
  3. board       : rows = the 6 souvenirs in List_no order,
                   columns = the 21 pilots in Pilot_no order,
                   cell = 1 iff that pilot bought that souvenir at a fair price
                   at least once ("HR checks which pilot brought a given
                   souvenir for a fair price").
  4. The 6x21 grid is a pixel image.  Read with 3-wide glyphs separated by
     blank columns -> the board spells the flag word.

Usage:
    /home/kozak/mllm/py solve.py [DATA_DIR] [OUT_DIR]
Defaults: DATA_DIR=../../tasks/task_spain/data (relative to this file),
          OUT_DIR=this directory.
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
```

---

## 4. 🏆 Результат и верификация
- **Итоговый флаг:** `SIGMOID_CRUCIO`
- **Результат проверки платформы:** Статус: passed, 100/100 баллов, Submission ID: 80.

---

## 5. 📂 Файлы решения в этой папке
- [`answer.txt`](./answer.txt)
- [`solution.md`](./solution.md)
- [`solve.py`](./solve.py)

---

## 🚀 Запуск решения
```bash
python solve.py
```
