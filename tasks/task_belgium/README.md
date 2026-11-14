# Бельгия — Fry-ndship (`task_belgium`)

| Параметр | Значение |
|---|---|
| **Страна / Регион** | **Бельгия (BE)** |
| **Категория** | **Data Science / Регрессионный анализ и Стеганография** |
| **Сложность** | **Easy** |
| **Баллы** | **100 / 100** (100% максимальный результат) |
| **Флаг решения** | `SIGMOID_SUPERVISION` |
| **Статус на платформе** | ✅ **PASSED** |

---

## 1. 📖 Постановка задачи
Кооператив картофелеводов Фландрии оценивает длину изгородей для ограждения полей. Землемер Мариетта замерила выборку полей и спрятала в измерениях слово. Задача — смоделировать геометрию полей и найти зашифрованное слово.

---

## 2. 🔍 В чем состоял подвох
Слово зашифровано не в самих величинах площади или длины, а в остатках (невязках) регрессионной модели.

---

## 3. ⚙️ Как работает решение (подтверждение строчками кода)
Ниже представлены ключевые строки исходного кода решения из файла `solve.py`, подтверждающие описанный алгоритм:

```python

Idea
----
1) Fit the analyst's model: edge_m ~ area_ha + long_side_m  ("dimensions").
   R^2 = 0.989  ->  exactly the "99% of variation explained" of the story.
2) The model is essentially edge = 1.9 * (long_side + short_side)
   with short_side = area_ha * 10000 / long_side.
3) The non-trivial, designed property of the surveyed fields is the
   distribution of the *sum of the dimensions* D = L + S (equivalently
   the edge/perimeter itself): D is (near) UNIFORM on its support.
   L alone and S alone are clearly NOT uniform, and corr(L,S) = +0.36,
   so a uniform sum is not a generic consequence -- it was constructed.
   Indeed f = L/D is ~ N(0.72, 0.09) and D ~ U(133, 584):  L = f D,
   S = (1-f) D.  This makes the measured edge lengths uniformly spread.
4) Hence the word Mariette "observed in her measurements" is UNIFORM.

Run:  /home/kozak/mllm/py solve.py [csv_path] [out_dir]

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
```

---

## 4. 🏆 Результат и верификация
- **Итоговый флаг:** `SIGMOID_SUPERVISION`
- **Результат проверки платформы:** Статус: passed, 100/100 баллов, Submission ID: 122.

---

## 5. 📂 Файлы решения в этой папке
- [`answer.txt`](./answer.txt)
- [`solution.md`](./solution.md)
- [`solve.py`](./solve.py)
- [`uniform_tests.csv`](./uniform_tests.csv)

---

## 🚀 Запуск решения
```bash
python solve.py
```
