# Мексика — Ruta de las Identidades (`task_mexico`)

| Параметр | Значение |
|---|---|
| **Страна / Регион** | **Мексика (MX)** |
| **Категория** | **Data Engineering / Разрешение сущностей (Entity Resolution)** |
| **Сложность** | **Easy** |
| **Баллы** | **100 / 100** (100% максимальный результат) |
| **Флаг решения** | `SIGMOID_RECORD_LINKAGE` |
| **Статус на платформе** | ✅ **PASSED** |

---

## 1. 📖 Постановка задачи
Программа 'Ruta de Lectura' объединила три таблицы читателей (19 520 записей). Из-за опечаток, сокращений и перестановок одни и те же люди встречаются несколько раз. Нужно найти все уникальные пары дубликатов.

---

## 2. 🔍 В чем состоял подвох
Наивное попарное сравнение 19 520 записей требует ~190 млн операций, что не укладывается в тайм-аут. При этом имена содержат перестановки имени и фамилии, инициалы, а адреса — мексиканские сокращения.

---

## 3. ⚙️ Как работает решение (подтверждение строчками кода)
Ниже представлены ключевые строки исходного кода решения из файла `main.py`, подтверждающие описанный алгоритм:

```python

Entity resolution over three merged registration tables.  Each row has
``source_table, record_id, first_name, last_name, address`` and the task is to
emit every unordered pair of record ids that belongs to the same person, as
``{"matches": [[id_a, id_b], ...]}`` (all pairs of a cluster, e.g. 3 pairs for
a cluster of 3).

How it is run
-------------
The grader copies this file into a scratch directory together with the hidden
``test-input.csv`` and runs ``python main.py`` (same convention as the other
LMML file tasks).  This script therefore:

* finds the input CSV from argv / environment / ``test-input.csv`` /
  ``train-input.csv`` / any CSV with the required header,
* writes ``test-output.json`` next to the working directory (and next to the
  input file), and also prints the same JSON object on stdout,
* exposes ``solve(csv_path) -> dict`` for import-based harnesses.

Method
------
The address is the stable identifier: after canonicalisation (accents,
punctuation, component re-ordering, and the register's abbreviation table such
as ``C.jón`` -> ``callejon``, ``Int.`` -> ``interior``) the records of one
person carry the *same multiset* of address tokens, while different people
```

---

## 4. 🏆 Результат и верификация
- **Итоговый флаг:** `SIGMOID_RECORD_LINKAGE`
- **Результат проверки платформы:** Статус: passed, 100/100 баллов, PASS: all 12077 matching pairs.

---

## 5. 📂 Файлы решения в этой папке
- [`answer.txt`](./answer.txt)
- [`evidence.txt`](./evidence.txt)
- [`main.py`](./main.py)
- [`solution.md`](./solution.md)
- [`stress_test.py`](./stress_test.py)
- [`validate.py`](./validate.py)

---

## 🚀 Запуск решения
```bash
python main.py
```
