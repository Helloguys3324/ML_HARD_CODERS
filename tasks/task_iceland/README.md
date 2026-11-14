# Исландия — What The Data? (`task_iceland`)

| Параметр | Значение |
|---|---|
| **Страна / Регион** | **Исландия (IS)** |
| **Категория** | **Data Engineering / Очистка данных и Дедупликация** |
| **Сложность** | **Easy** |
| **Баллы** | **100 / 100** (100% максимальный результат) |
| **Флаг решения** | `SIGMOID_PREPROCESSING` |
| **Статус на платформе** | ✅ **PASSED** |

---

## 1. 📖 Постановка задачи
Метеобюро Исландии собирает данные геотермальных источников через старый спутниковый канал. Экспорт поврежден: повторные спутниковые пересылки, исландские буквы повреждены кодировкой (модзибакэ), температуры передаются в Фаренгейтах и Цельсиях.

---

## 2. 🔍 В чем состоял подвох
Грейдер платформы требует побайтового совпадения всех колонок и строгого порядка строк по reading_id.

---

## 3. ⚙️ Как работает решение (подтверждение строчками кода)
Ниже представлены ключевые строки исходного кода решения из файла `main.py`, подтверждающие описанный алгоритм:

```python

Reads ``test-input.csv`` from the current working directory (bare filename)
and writes ``test-output.csv`` next to it.

Damage handled generically:
  * legacy preamble: comment lines starting with '#' and blank lines, any length
  * ';' delimiter, decimal comma or decimal point in temperatures
  * retransmission duplicates: one output row per reading_id
  * mojibake station names (UTF-8 bytes mis-decoded through cp1252/latin-1),
    possibly several levels deep -> repair until stable
  * Fahrenheit readings -> Celsius, exactly one decimal place
  * rows sorted by numeric reading_id ascending

Standard library only.

import csv
import sys
import unicodedata
from collections import Counter

INPUT_NAME = "test-input.csv"
OUTPUT_NAME = "test-output.csv"

EXPECTED_COLUMNS = ["reading_id", "station_name", "temperature_raw", "unit", "reading_date"]

```

---

## 4. 🏆 Результат и верификация
- **Итоговый флаг:** `SIGMOID_PREPROCESSING`
- **Результат проверки платформы:** Статус: passed, 100/100 баллов, 80 из 80 строк закрытого теста совпали идеально.

---

## 5. 📂 Файлы решения в этой папке
- [`main.py`](./main.py)
- [`solution.md`](./solution.md)
- [`stress_test.py`](./stress_test.py)

---

## 🚀 Запуск решения
```bash
python main.py
```
