# Отчёт 22: task_cote_divoire — катастрофическое забывание и rehearsal

**Дата:** 14 ноября 2026, 16:05 EET
**Задача:** `task_cote_divoire` («The Abidjan Transit Ledger»), Medium, 200 баллов
**Флаг:** `SIGMOID_REHEARSAL`
**Статус:** `passed`, 200/200, с первого сабмита
**Счёт команды:** 5200 (31/54)

---

## 1. Диагноз: не backdoor, а catastrophic forgetting

| Модель | old-task | new-task |
|---|---|---|
| `original_model.joblib` | 1.00 | 0.00 |
| `finetuned_model.joblib` | 0.00 | 1.00 |

Пространство меток сменилось целиком: 5 транзакционных SQL-шаблонов → 5 **других** spatial/window-шаблонов. Все старые decision-регионы перезаписаны, поэтому роутер уверенно отправляет «insert an order» в spatial-шаблон.

Название флага прямо указывает на метод лечения — **rehearsal** (replay старой задачи).

## 2. Восстановление

```text
FeatureUnion(word tf-idf 1-2 grams, char_wb 3-5 grams) → LinearSVC
обучение: 279 уникальных примеров (train-*.json 250 + old_task_sample.json 40)
```

- 5-fold CV: old/new = **1.000/1.000**
- holdout на 40 old-сэмплах: 40/40
- 10 рукописных невиданных перефразировок: 10/10
- переобучение < 1 с на CPU

Сабмит — самодостаточный `main.py` (данные вшиты gzip+base64, переобучается на лету).

## 3. Ответ платформы

```text
old-task accuracy: 0.9733 (threshold 0.85) → PASS
new-task accuracy: 0.9867 (threshold 0.85) → PASS
BOTH GATES PASSED. FLAG: SIGMOID_REHEARSAL
```

## 4. Замечание

Чекпоинты из описания (`original/finetuned_model.joblib`, `vectorizer.joblib`) в Drive-папке отсутствуют — там только 6 файлов данных. Модель обучена с нуля, что полностью покрывает оба порога.

Артефакты: `challenges/task_cote_divoire/{solve.py,build_main.py,main.py,eval_cv.py,writeup.md,checkpoints/}`.