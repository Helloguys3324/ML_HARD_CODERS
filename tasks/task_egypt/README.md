# Египет — Always Valid, No Excuses (`task_egypt`)

| Параметр | Значение |
|---|---|
| **Страна / Регион** | **Египет (EG)** |
| **Категория** | **Data Engineering / Нормализация OCR в строгий JSON** |
| **Сложность** | **Easy** |
| **Баллы** | **100 / 100** (100% максимальный результат) |
| **Флаг решения** | `SIGMOID_PHARAOH` |
| **Статус на платформе** | ✅ **PASSED** |

---

## 1. 📖 Постановка задачи
Распознанные OCR карточки музейных экспонатов Каира содержат поля с разделителем `|`, оборванные заметки, маркеры страниц и повреждения. Требовалось преобразовать каждую карточку в строго валидный однострочный JSON без потери данных.

---

## 2. 🔍 В чем состоял подвох
Строгие правила типизации: сегменты без двоеточия являются шумом OCR и отбрасываются; пустые значения становятся null; числа парсятся в int/float; двоеточия внутри значений (например, 'role: architect, vizier: and healer') не должны разбивать значение.

---

## 3. ⚙️ Как работает решение (подтверждение строчками кода)
Ниже представлены ключевые строки исходного кода решения из файла `main.py`, подтверждающие описанный алгоритм:

```python
# -*- coding: utf-8 -*-
"Always Valid, No Excuses" - OCR record cards -> one strict JSON line each.

Input : test-input.txt  (UTF-8, one OCR record per line, fields `key: value`
        separated by `|`, messy spacing, interspersed scan noise).
Output: test-output.txt (one strict JSON object per input line, nothing added).

Approach
--------
The provided local oracle (Qwen3-1.7B-GGUF) is a *generator*, not a validator:
left alone it quotes numbers, drops fields and mangles backslashes.  So the
pipeline is an oracle-in-the-loop that is held to the format:

  1. A deterministic rule engine parses the raw OCR card exactly as the task
     spec defines: split on the first `:` only, trim key/value, keep only
     segments whose key is a single simple word (letters only), drop every
     other segment (page numbers, chapter headers, footers, bracketed marks,
     figure labels, garble) and type the value (null / bool / int / string).
  2. The optional local model (loaded from models/*.gguf next to this file,
     via llama-cpp-python) is consulted only for records the rule engine
     cannot resolve on its own.  Its answer is re-parsed and re-normalised by
     the same rule engine, so oracle drift can never leak into the output.
     If the model is absent (the normal offline/evaluation case) the rule
     engine alone produces the output; the model is never required.
  3. Every emitted line is round-tripped through json.loads() before it is
```

---

## 4. 🏆 Результат и верификация
- **Итоговый флаг:** `SIGMOID_PHARAOH`
- **Результат проверки платформы:** Статус: passed, 100/100 баллов, PASS: 100/100.

---

## 5. 📂 Файлы решения в этой папке
- [`answer.txt`](./answer.txt)
- [`main.py`](./main.py)
- [`main.py.txt`](./main.py.txt)
- [`proto.py`](./proto.py)
- [`solution.md`](./solution.md)
- [`test-output.txt`](./test-output.txt)
- [`validate.py`](./validate.py)

---

## 🚀 Запуск решения
```bash
python main.py
```
