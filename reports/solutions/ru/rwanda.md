# Отчёт о решении: task_rwanda ("The Drying Floor at Karongi")

## 1. Обзор и цели задачи
- **Task ID:** `task_rwanda` (Руанда).
- **Сложность и баллы:** Easy, 100 / 100 баллов.
- **Область:** General AI / Конечные автоматы агентов (LangGraph).
- **Формат вывода:** Исправление бага в `graph.py`, открывающее флаг `SIGMOID_<NAME>`.
- **Постановка задачи:**
  - Сельскохозяйственный кофейный кооператив в Каронги (озеро Киву) автоматизировал отслеживание сушки зёрен с помощью конечного автомата на базе LangGraph:
    $$\text{receive\_batch} \longrightarrow \text{inspect\_batch} \longrightarrow (\text{готов?}) \longrightarrow \begin{cases} \text{dry\_batch} \to \text{inspect\_batch} \to \dots & (\text{не одобрен}) \\ \text{grade\_batch} \to \text{END} & (\text{одобрен}) \end{cases}$$
  - В исходной кодовой базе процесс зацикливался. Поступающие партии кофе, даже находящиеся в пределах допустимой влажности, бесконечно циркулировали между инспекцией и сушкой вплоть до превышения лимита рекурсии (`GraphRecursionError`).
  - Цель — диагностировать и устранить ошибку в определении графа состояний.

---

## 2. Ключевые идеи и анализ первопричины
- **Механизм состояний графа:**
  - Рабочий процесс исполняется поверх `StateGraph(BatchState)`, где атрибуты состояния определяются через `TypedDict`.
  - Условная маршрутизация после замера влажности задаётся функцией:
    ```python
    def route_after_inspection(state: BatchState) -> Literal["dry_batch", "grade_batch"]:
        if state["approved"]:
            return "grade_batch"
        return "dry_batch"
    ```
- **Локализация первопричины (`graph.py`):**
  - В исходном коде графа ребро из узла `inspect_batch` было задано как безусловное прямое ребро:
    ```python
    # Некорректное соединение узлов:
    builder.add_edge("inspect_batch", "dry_batch")
    ```
  - Из-за отсутствия вызова `builder.add_conditional_edges(...)` для `inspect_batch`, функция-маршрутизатор `route_after_inspection` полностью игнорировалась в рантайме.
  - Каждая партия кофе безусловно отправлялась обратно в `dry_batch` независимо от флага готовности или фактической влажности.

---

## 3. Детали реализации
Исправление внесено напрямую в `graph.py`:

1. **Замена некорректного ребра:**
   - Удалено жёстко закодированное безусловное ребро `builder.add_edge("inspect_batch", "dry_batch")`.
   - Вместо него настроена условная маршрутизация:
     ```python
     builder.add_conditional_edges(
         "inspect_batch",
         route_after_inspection,
         {
             "dry_batch": "dry_batch",
             "grade_batch": "grade_batch",
         },
     )
     ```
2. **Гарантия завершения процесса:**
   - Подтверждено, что узел `grade_batch` корректно переходит в терминальное состояние `END`:
     ```python
     builder.add_edge("grade_batch", END)
     ```
3. **Инварианты границ влажности:**
   - Проверено, что `inspect_batch` корректно оценивает допустимый диапазон влажности:
     ```python
     def inspect_batch(state: BatchState) -> dict:
         ready = MOISTURE_MIN <= state["moisture"] <= MOISTURE_MAX
         return {"inspections": state["inspections"] + 1, "approved": ready}
     ```
4. **Верификация выполнения:**
   - Проверено, что партии с начальной соответствующей влажностью (например, 12.0%) напрямую направляются в `grade_batch` и завершаются в `END` за один проход без повторов.
   - Партии с превышением влажности циклически проходят `dry_batch` до входа в интервал $[10.5\%, 12.5\%]$, надёжно завершая работу.

---

## 4. Результаты и артефакты
- **Статус:** PASSED
- **Баллы:** 100 / 100 баллов
- **Восстановленный флаг:** `SIGMOID_CONVERGENCE`
- **Модифицированный файл:** `graph.py`
