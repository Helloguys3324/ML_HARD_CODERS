# Сенегал — The Teranga FM Token Diet (`task_senegal`)

| Параметр | Значение |
|---|---|
| **Страна / Регион** | **Сенегал (SN)** |
| **Категория** | **NLP / Оптимизация промптов и LLM-шлюзов** |
| **Сложность** | **Medium** |
| **Баллы** | **200 / 200** (100% максимальный результат) |
| **Флаг решения** | `SIGMOID_TOKENDIET` |
| **Статус на платформе** | ✅ **PASSED** |

---

## 1. 📖 Постановка задачи
Офлайн-симулятор LLM-шлюза Teranga FM (`model_utils.py`) обрабатывает 20 фиксированных вопросов. Требуется подобрать конфигурацию запросов так, чтобы все 20 ответов прошли проверку качества, а суммарный расход токенов не превысил 1200.

---

## 2. 🔍 В чем состоял подвох
Симулятор возвращает целевые слоты только при выполнении логического условия `(assisted or not large_only)`, где `assisted = (tier=='large') or context`.

---

## 3. ⚙️ Как работает решение (подтверждение кодом)
1. Разобрали логику генерации ответов в `model_utils.py`.
2. Выявили, что флаг `fewshot` тратит лишние токены и не нужен для правильного ответа, а стиль `terse` минимален по расходу.
3. Провели перебор всех 16 конфигураций для каждого из 20 вопросов:
   - вопросы без слотов `large_only`: режим `cheap / terse / context=0`;
   - с `large_only` и коротким контекстом: `cheap + context=true`;
   - с `large_only` и длинным контекстом: `large / terse / context=0`.
4. Достигли абсолютного минимума в **1160 токенов** из 1200 (экономия 68%), пройдя 20 из 20 проверок качества.

### Ключевой фрагмент кода решения:
```python
# Из solve.py: глобальная оптимизация диеты токенов симулятора LLM-шлюза
# Completion содержит слот только если (assisted or not large_only), assisted = (tier=='large') or context
config = optimize_routing(questions, fact_bank, max_tokens=1200)
# Достигнут расход 1160 / 1200 токенов (экономия 68%) при 20/20 точности -> SIGMOID_TOKENDIET
```

---

## 4. 🏆 Результат и верификация
- **Итоговый флаг:** `SIGMOID_TOKENDIET`
- **Результат проверки платформы:** Статус: passed, 200/200 баллов, 1160 токенов, Submission ID: 1041.

---

## 5. 📂 Файлы решения в этой папке
- [`answers.json`](./answers.json)
- [`example_naive_config.json`](./example_naive_config.json)
- [`fact_bank.json`](./fact_bank.json)
- [`main.py`](./main.py)
- [`model_utils.py`](./model_utils.py)
- [`routing_config.json`](./routing_config.json)
- [`solve.py`](./solve.py)
- [`submission.json`](./submission.json)
- [`writeup.md`](./writeup.md)

---

## 🚀 Запуск решения
```bash
python solve.py
```
