# Отчёт 11: Решена task_switzerland — 900 очков, HardCoders удерживают 1-е место!

**Дата и время:** 14 ноября 2026, 12:45 EET  
**Задача:** `task_switzerland` (The Weak Layer) — Швейцария, Европа  
**Статус:** `passed` (100 / 100 баллов)  
**ID сабмита:** 161  
**Общий счёт HardCoders:** **900 очков (1-е место в мире)**  
*Примечание: сокомандник Veaceslav Coltuc также успешно сдал `task_nigeria` (сабмит 157, 100 очков)!*

---

## 1. Логика решения задачи
- В данных `train-input.csv` (6000 наблюдений снежных шурфов в Давосе) обучается классификатор `DecisionTreeClassifier(random_state=42)`.
- Скрытое слово закодировано в порядке вопросов, ведущих к сходу лавины (`slab_released == 1`):
  1. **D** — `depth_hoar_pct`
  2. **E** — `elevation_m`
  3. **M** — `mean_grain_roundness`
  4. **O** — `overburden_load_kpa`
  $\implies$ Слово: `DEMO`.
- Создан автономный скрипт `challenges/task_switzerland/main.py`, который динамически считывает `test-input.csv`, обучает дерево решений, извлекает слово по цепочке сплитов и записывает флаг в `test-output.txt`.
- Сабмит принят сервером со 100% результатом (`test-output.txt matches the expected flag`).
