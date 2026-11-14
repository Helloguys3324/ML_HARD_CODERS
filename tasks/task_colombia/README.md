# Колумбия — Looks Good to Me! — Bogotá Sound Check (`task_colombia`)

| Параметр | Значение |
|---|---|
| **Страна / Регион** | **Колумбия (CO)** |
| **Категория** | **Безопасность данных / Форензика и Санитаризация** |
| **Сложность** | **Easy** |
| **Баллы** | **100 / 100** (100% максимальный результат) |
| **Флаг решения** | `SIGMOID_DATA_SANITIZATION` |
| **Статус на платформе** | ✅ **PASSED** |

---

## 1. 📖 Постановка задачи
Музыкальная школа в Боготе отбирает инструменты по отзывам студентов. В архив закрались измененные отзывы. К отзывам прикреплены цифровые печати. Идентификаторы измененных записей открывают скрытое сообщение.

---

## 2. 🔍 В чем состоял подвох
Изменения выполнены тремя разными способами: визуальные омоглифы, невидимые zero-width пробелы и переворот оценок. Пропуск хотя бы одного из трех типов фальсификации не дает правильного хеша.

---

## 3. ⚙️ Как работает решение (подтверждение строчками кода)
Ниже представлены ключевые строки исходного кода решения из файла `solve.py`, подтверждающие описанный алгоритм:

```python
task_colombia — "Looks Good to Me! — Bogotá Sound Check"

Reproducible analysis of colombia_flag_data.zip / reviews.json.

What this script does
---------------------
1. Extracts reviews.json into ./data (never touches the original zip).
2. Scans `trusted_reference` + `samples` for character-level tampering
   (invisible / homoglyph characters) and shows the records that contain them.
3. Runs near-duplicate detection between `samples` and `trusted_reference`
   (character n-gram TF-IDF + difflib) and prints the strongest pairs.
4. Implements the archive_note procedure:
       mask = sha256( ",".join(sorted(sample_index)) )
       plaintext = archive_tag_bytes XOR mask, rstrip(b"\x00")
   and tests the candidate altered-record sets (invisible-char records,
   near-duplicate records, and their union) against every archive_tag.
5. Prints diagnostics: how many tags decode to printable text, longest
   trailing-zero run, etc.

The script intentionally does NOT submit anything.  Run with:
    /home/kozak/mllm/py work/solutions/task_colombia/solve.py

from __future__ import annotations

import hashlib
```

---

## 4. 🏆 Результат и верификация
- **Итоговый флаг:** `SIGMOID_DATA_SANITIZATION`
- **Результат проверки платформы:** Статус: passed, 100/100 баллов, Submission ID: 108.

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
