# Германия — Black Box (`task_germany`)

| Параметр | Значение |
|---|---|
| **Страна / Регион** | **Германия (DE)** |
| **Категория** | **Активное обучение / Оптимизация черного ящика** |
| **Сложность** | **Hard** |
| **Баллы** | **300 / 300** (100% максимальный результат) |
| **Флаг решения** | `SIGMOID_FACTORIAL` |
| **Статус на платформе** | ✅ **PASSED** |

---

## 1. 📖 Постановка задачи
Скрытая детерминированная функция `f(x1..x5)` на единичном гиперкубе `[0, 1]^5`. Требуется за не более чем 150 запросов к оракулу полностью восстановить структуру функции (активные входы, ступени и значения).

---

## 2. 🔍 В чем состоял подвох
Случайный поиск или равномерная сетка требуют тысяч запросов при бюджете всего в 150.

---

## 3. ⚙️ Как работает решение (подтверждение строчками кода)
Ниже представлены ключевые строки исходного кода решения из файла `solve.py`, подтверждающие описанный алгоритм:

```python
Recover the Black Box — structure discovery solver (task_germany).

Hidden noise-free deterministic function family on [0,1]^5:

    f(x) = C + sum_i a_i*x_i
             + c*(x_p - 0.5)*(x_q - 0.5)      # exactly one centred-product interaction
             + J*[ g . x >= T ]               # exactly one linear step (jump J)

Approach (robust to the one-variable-at-a-time trap *and* to a step that
overlaps the smooth terms, because it never relies on a global smooth fit):

  Phase 1  finite-difference stencil (small h, several base points).
           Mixed second differences -> interaction pair + coefficient c;
           corrected forward differences -> linear coefficients a_i;
           both via the MEDIAN over base points, so a stencil that happens to
           straddle the step cannot corrupt the estimate.
  Phase 2  with the smooth part known, the residual r = f - smooth is exactly
           0 (off-step) or J (on-step): the step pops out cleanly. Corners are
           labelled jump/base, the gate SUPPORT is read by scanning cube edges.
  Phase 3  the switching hyperplane is reconstructed by bisection strictly
           inside the gate subspace (idle coords fixed to 0) and a PCA plane fit.
  Phase 4  the normal is oriented onto the jump side; the smooth coefficients
           are refit exactly on off-step points; predictions are produced.

Only numpy is used beyond the stdlib.
```

---

## 4. 🏆 Результат и верификация
- **Итоговый флаг:** `SIGMOID_FACTORIAL`
- **Результат проверки платформы:** Статус: passed, 300/300 баллов, 140/150 запросов.

---

## 5. 📂 Файлы решения в этой папке
- [`dev_test.py`](./dev_test.py)
- [`main.py`](./main.py)
- [`solve.py`](./solve.py)

---

## 🚀 Запуск решения
```bash
python solve.py
```
