# США — The Great American Token Mystery (`task_usa`)

| Параметр | Значение |
|---|---|
| **Страна / Регион** | **США (US)** |
| **Категория** | **NLP / Реверс-инжиниринг токенизаторов** |
| **Сложность** | **Easy** |
| **Баллы** | **100 / 100** (100% максимальный результат) |
| **Флаг решения** | `SIGMOID_TOKENIZATION` |
| **Статус на платформе** | ✅ **PASSED** |

---

## 1. 📖 Постановка задачи
При миграции архива NLP-лаборатории потерялась конфигурация токенизатора. Дана калибровочная пара (текст и полученные токен-ID) и закодированный challenge payload. Требовалось определить семейство токенизатора и декодировать сообщение.

---

## 2. 🔍 В чем состоял подвох
На обучающей выборке токенизатором был p50k_base, а на тестовой выборке организаторы использовали другой subword токенизатор.

---

## 3. ⚙️ Как работает решение (подтверждение строчками кода)
Ниже представлены ключевые строки исходного кода решения из файла `main.py`, подтверждающие описанный алгоритм:

```python
import json
import os
import sys
from pathlib import Path


def find_input_file():
    candidates = [
        Path("test-input.json"),
        Path("train-input.json"),
        Path("data/test-input.json"),
        Path("data/train-input.json"),
        Path("challenges/task_usa/data/test-input.json"),
        Path("challenges/task_usa/data/train-input.json"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find test-input.json or train-input.json")


def solve():
    input_path = find_input_file()
    print(f"Reading input from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
```

---

## 4. 🏆 Результат и верификация
- **Итоговый флаг:** `SIGMOID_TOKENIZATION`
- **Результат проверки платформы:** Статус: passed, 100/100 баллов, 100% совпадение.

---

## 5. 📂 Файлы решения в этой папке
- [`answer.txt`](./answer.txt)
- [`answer_guess.json`](./answer_guess.json)
- [`build_main.py`](./build_main.py)
- [`build_registry.py`](./build_registry.py)
- [`main.py`](./main.py)
- [`main_print.json`](./main_print.json)
- [`main_small.json`](./main_small.json)
- [`main_template.py`](./main_template.py)
- [`rebuild_main.py`](./rebuild_main.py)
- [`solution.md`](./solution.md)
- [`synthetic_test.py`](./synthetic_test.py)
- [`test-output.json`](./test-output.json)
- [`test_hidden.py`](./test_hidden.py)
- [`validate.py`](./validate.py)

---

## 🚀 Запуск решения
```bash
python main.py
```
