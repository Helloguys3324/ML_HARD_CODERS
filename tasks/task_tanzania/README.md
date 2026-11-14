# Танзания — Please, You Are So Delusional! (`task_tanzania`)

| Параметр | Значение |
|---|---|
| **Страна / Регион** | **Танзания (TZ)** |
| **Категория** | **NLP / Информационный поиск (RAG / BM25)** |
| **Сложность** | **Easy** |
| **Баллы** | **100 / 100** (100% максимальный результат) |
| **Флаг решения** | `SIGMOID_A69EE28C5B2214A9AD99` |
| **Статус на платформе** | ✅ **PASSED** |

---

## 1. 📖 Постановка задачи
Программа передвижных научных выставок 'Safari ya Sayansi' использует цифрового ассистента для навигации посетителей между городами Дар-эс-Салам, Аруша и Мванза. Однако ретривер часто отправляет посетителей к устаревшим архивным бюллетеням вместо актуальных инструкций доступа.

---

## 2. 🔍 В чем состоял подвох
В стартовом коде `retrieval_pipeline.py` заложены две критические ошибки: (1) векторные пространства запроса и документов не согласованы — документы кодировались через all-MiniLM-L6-v2, а вопросы через e5-small-v2; (2) наивный чанкинг по 180 символов разрезал заголовки и названия залов. Кроме того, в корпусе намеренно размещены 72 архивные ловушки с фразой 'does not replace the current access notice', которые путали наивный поиск.

---

## 3. ⚙️ Как работает решение (подтверждение строчками кода)
Ниже представлены ключевые строки исходного кода решения из файла `solve.py`, подтверждающие описанный алгоритм:

```python

This script is a lightweight verifier: it re-derives the flag from the repaired
``retrieved_ids.json`` and independently validates every retrieved document
against the question text (exhibit + city -> "CURRENT ACCESS NOTICE: ...").

Usage:
    /home/kozak/mllm/py solve.py [--data work] [--results retrieved_ids.json]

Use ``run_pipeline.sh`` (or run ``retrieval_pipeline.py`` directly) to
regenerate ``retrieved_ids.json`` from scratch with the repaired retriever.

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from flag_utils import derive_flag_from_files, load_verification_codes

CITIES = ("Dar es Salaam", "Arusha", "Mwanza")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
```

---

## 4. 🏆 Результат и верификация
- **Итоговый флаг:** `SIGMOID_A69EE28C5B2214A9AD99`
- **Результат проверки платформы:** Статус: passed, 100/100 баллов, Submission ID: 86.

---

## 5. 📂 Файлы решения в этой папке
- [`answer.txt`](./answer.txt)
- [`diagnostics.json`](./diagnostics.json)
- [`flag_utils.py`](./flag_utils.py)
- [`retrieval_evidence.csv`](./retrieval_evidence.csv)
- [`retrieval_pipeline.py`](./retrieval_pipeline.py)
- [`retrieved_ids.json`](./retrieved_ids.json)
- [`run_pipeline.sh`](./run_pipeline.sh)
- [`solution.md`](./solution.md)
- [`solve.py`](./solve.py)

---

## 🚀 Запуск решения
```bash
python solve.py
```
