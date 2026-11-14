# Отчёт 21: task_germany и task_senegal — 5000 очков, 30/54

**Дата:** 14 ноября 2026, 15:50 EET
**Счёт команды:** **5000** (2-е место; лидер SeKar — 5100, отставание 100)
**Сдано задач:** 30 из 54

---

## 1. `task_germany` («Black Box») — Hard, 300, `SIGMOID_FACTORIAL`

Восстановление структуры скрытой функции `f(x1..x5)` на `[0,1]^5` бюджетом ≤150 запросов.

- Сдано: `queries 140/150`, `score 100.0/100`.
- Подход: скрининг влияющих координат (дисперсия при вариации одной оси), затем бинарный поиск порогов по активным осям и проверка взаимодействий вторыми разностями.
- Ответ платформы: `FLAG: SIGMOID_FACTORIAL`.

## 2. `task_senegal` («The Teranga FM Token Diet») — Medium, 200, `SIGMOID_TOKENDIET`

Оптимизация «диеты токенов» на прозрачном офлайн-симуляторе шлюза (`model_utils.py`).

Ключевой факт симулятора:
```text
completion содержит phrasings[0] слота ТОЛЬКО если (assisted or not large_only),
где assisted = (tier == "large") or context
```
Качество проверяется substring-сканом без частичного кредита.

Стратегия по 4 ручкам (`tier`, `verbose/terse`, `context`, `fewshot`):

| Класс вопросов | Оптимум |
|---|---|
| без `large_only`-слотов (8) | `cheap / terse / context=0` |
| `large_only` + короткий сниппет (6) | `cheap + context=true` (RAG дешевле большого тира) |
| `large_only` + длинный сниппет (6) | `large / terse / context=0` |

`fewshot` всегда `false`. Полный перебор 16 конфигураций × 20 вопросов дал глобальный минимум.

Результат платформы: `quality: 20/20`, `tokens: 1160 / 1200` → PASS, `FLAG: SIGMOID_TOKENDIET`.
Экономия: 3626 → 1160 токенов (−68%).

## 3. Уточнение по `task_nigeria`

Флаг, сданный сокомандником: `SIGMOID_TALKINGDRUM`.

## 4. Статус гонки

```text
1. SeKar        5100
2. HardCoders   5000   ← −100
3. Devian       2500
```

До первого места — одна Easy-задача (`france` / `india` / `laos`, по 100).