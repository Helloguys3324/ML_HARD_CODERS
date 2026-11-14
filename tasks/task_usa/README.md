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

## 3. ⚙️ Как работает решение (подтверждение кодом)
1. Протестировали все словари `tiktoken` (cl100k_base, o200k_base, p50k_base, r50k_base) и библиотеки HuggingFace.
2. Точное совпадение токенов калибровочного текста подтвердило словарь **cl100k_base**.
3. Декодировали целевую последовательность токенов через найденный токенизатор и извлекли валидное сообщение с флагом `SIGMOID_TOKENIZATION`.

### Ключевой фрагмент кода решения:
```python
# Из main.py: верификация токенизатора cl100k_base по калибровочной паре
enc = tiktoken.get_encoding("cl100k_base")
assert enc.encode(calibration_text) == calibration_token_ids
decoded_message = enc.decode(challenge_token_ids)
# FLAG: SIGMOID_TOKENIZATION
```

---

## 4. 🏆 Результат и верификация
- **Итоговый флаг:** `SIGMOID_TOKENIZATION`
- **Результат проверки платформы:** Статус: passed, 100/100 баллов, 100% совпадение.

---

## 5. 📂 Файлы решения в этой папке
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
