# task_senegal — «The Teranga FM Token Diet» — solve writeup

**Результат:** `PASS` (quality 20/20, budget 1160/1200) · флаг `SIGMOID_TOKENDIET`

## Суть
Teranga FM гоняет 20 фиксированных вопросов через LLM-шлюз, который биллит
токены за prompt и completion. Нужно оставить все 20 ответов корректными,
уложив суммарный счёт в `TOKEN_BUDGET = 1200`.

Механизм полностью детерминирован и открыт — `model_utils.py` + `fact_bank.json`.

## Как устроен симулятор (ключевые факты из `model_utils.py`)
Каждый вопрос имеет `slots` — требуемые факты с `phrasings` и флагом `large_only`.

`render_completion_body(slots, tier, style, context)` **(строка 103)**:
```python
assisted = tier == "large" or context
included = [s["phrasings"][0] for s in slots if assisted or not s["large_only"]]
```
Акт: в ответ (completion) попадает `phrasings[0]` слота **тогда и только тогда,
когда** `assisted` истинно **или** слот не `large_only`.

- `assisted` истинно при `tier == "large"` или `context == True` (при наличии
  `context_snippet`; для вопросов без сниппета `context` — no-op).
- Значит вопрос проходит качественный бар (`question_passes` — это тупой
  substring-скан каждого слота по `phrasings`) **детерминированно**, если
  включены все слоты. Опустить один — мгновенный fail, частичного кредита нет.
- `verbose` всего лишь добавляет INTRO+OUTRO вокруг фактов, `terse` — нет.
- `fewshot` добавляет фиксированный блок в prompt чисто «для уверенности» —
  на ответ не влияет вообще (убираем всегда).

Вывод: каждая комбинация из 4 ручек (`tier` ∈ {cheap,large}, `style` ∈
{verbose,terse}, `fewshot` ∈ {0,1}, `context` ∈ {0,1}) однозначно определяет и
стоимость, и факт прохождения. Для каждого вопроса берём **минимальную по
токенам** проходящую комбинацию.

## Правило маршрутизации (полный перебор по 16 комбинациям)
- **Вопрос без `large_only`-слотов** (`q01,q03,q09,q11,q14,q15,q16,q19`):
  `cheap / terse / fewshot=0 / context=0` — дешёвый тир и так включает все слоты
  (у него вообще нет `large_only`).
- **Вопрос с `large_only`-слотами** нужно «доукомплектовать». Два пути:
  1. `large`-тир (assisted=true) — платим длинным системным промптом;
  2. `cheap` + `context=true` — платим токенами сниппета (RAG-эффект).
  Берём тот, что дешевле **для данного вопроса** (зависит от длины сниппета vs
  длинного преамбула `large`).

Итог:
| группа | вопросы | маршрут |
|---|---|---|
| без `large_only` | q01 q03 q09 q11 q14 q15 q16 q19 | cheap тerse, context=0 |
| `large_only`, короткий сниппет | q02 q10 q13 q17 q18 q20 | cheap тerse, **context=1** |
| `large_only`, длинный сниппет | q04 q05 q06 q07 q08 q12 | **large** тerse, context=0 |

## Числа
- Наивная конфигурация интерна: **3626 токенов** (пролетает бюджет в 3 раза).
- Оптимальная: **1160 токенов** — все 20 вопросов проходят, бюджет-гейт `PASS`.
- Экономия: **2466 токенов (−68.0%)**.

```
TOTAL tokens = 1160 (budget 1200, cap delta 40)
quality_gate = True  budget_gate = True  passes = 20/20
```

## Файлы
- `solve.py` — полный перебор и самостоятельная проверка через `model_utils.run_config`.
- `routing_config.json` / `submission.json` — итоговый маршрут (один и тот же dict).
- `main.py` — копия `main_template.py`, I/O-контракт не тронут; `python main.py`
  читает `routing_config.json` и пишет `answers.json` (проверено: 20/20 PASS, 1160/1200).