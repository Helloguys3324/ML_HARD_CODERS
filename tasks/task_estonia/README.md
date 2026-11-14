# Эстония — This Data Is Perfect! (`task_estonia`)

| Параметр | Значение |
|---|---|
| **Страна / Регион** | **Эстония (EE)** |
| **Категория** | **Машинное обучение / Аудит утечки данных (Data Leakage)** |
| **Сложность** | **Medium** |
| **Баллы** | **200 / 200** (100% максимальный результат) |
| **Флаг решения** | `SIGMOID_77` |
| **Статус на платформе** | ✅ **PASSED** |

---

## 1. 📖 Постановка задачи
Система проверки заявок на получение e-Residency заявляла точность 93.42% при использовании модели 1-NN. Требовалось провести аудит пайплайна, найти истинную причину завышенного скора и вычислить реальную точность модели.

---

## 2. 🔍 В чем состоял подвох
Из трех подозрений (масштабирование до сплита, удачный сид разбиения, групповая утечка) первые два оказались ложными следами. Истинная утечка заключалась в том, что 1071 из 1971 заявок принадлежали одним и тем же 900 заявителям (пара дата рождения + гражданство).

---

## 3. ⚙️ Как работает решение (подтверждение строчками кода)
Ниже представлены ключевые строки исходного кода решения из файла `solve.py`, подтверждающие описанный алгоритм:

```python
Audit and Solution Write-Up for task_estonia (This Data Is Perfect!)
=====================================================================

1. The Real Bug and the Two Decoys
-----------------------------------
In the reported pipeline (`project_pipeline.py`), the fraud-detection team claimed
a 93.42% accuracy using a 1-Nearest-Neighbor classifier on 6 standardized numerical features.
The audit revealed three choices that look questionable at first glance:

- Decoy 1: Pre-split Feature Scaling (StandardScaler fitted on entire dataset)
  `X_scaled = StandardScaler().fit_transform(X)` before `train_test_split(...)`.
  Practitioners immediately flag this as textbook data leakage (test set distribution
  statistics leaking into the scaler).
  Evidence: Fitting StandardScaler strictly on the training fold yields 93.67% accuracy.
  The score remains virtually unchanged (+0.25%, a 1-sample difference on 395 test rows).
  This proves pre-split scaling is NOT what inflated the score.

- Decoy 2: Evaluation on a Single Random Train/Test Split (Evaluation Variance / Seed)
  `train_test_split(..., test_size=0.2, random_state=1137)` without cross-validation or stratification.
  One might hypothesize that random_state=1137 was cherry-picked or an artifact of an 80/20 split.
  Evidence: Running full row-level Leave-One-Out CV (`LeaveOneOut()`) across all 1971 rows
  yields 95.94% accuracy (and stratified split yields 93.92%). The score is still heavily
  inflated (~94-96%), proving the single split was NOT the root cause.

- The Real Bug: Identity Dropping Without Grouped Splitting (Group / Duplicate Identity Leakage)
```

---

## 4. 🏆 Результат и верификация
- **Итоговый флаг:** `SIGMOID_77`
- **Результат проверки платформы:** Статус: passed, 200/200 баллов, Submission ID: 410.

---

## 5. 📂 Файлы решения в этой папке
- [`solve.py`](./solve.py)

---

## 🚀 Запуск решения
```bash
python solve.py
```
