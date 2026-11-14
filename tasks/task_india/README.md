# Индия — Twins Separated at Birth (`task_india`)

| Параметр | Значение |
|---|---|
| **Страна / Регион** | **Индия (IN)** |
| **Категория** | **Компьютерное зрение / Поиск дубликатов изображений** |
| **Сложность** | **Easy** |
| **Баллы** | **100 / 100** (100% максимальный результат) |
| **Флаг решения** | `SIGMOID_VISUAL_ECHO` |
| **Статус на платформе** | ✅ **PASSED** |

---

## 1. 📖 Постановка задачи
В фотоархиве индийских памятников перемешаны истинные пары-дубликаты (одно исходное фото с разным сжатием, кропом или цветокоррекцией) и разные снимки одного и того же памятника с похожих ракурсов. Требуется найти все истинные пары.

---

## 2. 🔍 В чем состоял подвох
Разные фото одного монумента легко дают высокий глобальный цветовой скор, но имеют разное положение облаков, людей и перспективу.

---

## 3. ⚙️ Как работает решение (подтверждение строчками кода)
1. Построили быстрый скрининг кандидатов по перцептивным хешам (pHash + dHash).
2. Для пар-кандидатов рассчитали ключевые точки SIFT и сопоставили дескрипторы через FLANN с тестом Лоу (Lowe's ratio test <= 0.75).
3. Проверили геометрическую согласованность через RANSAC-гомографию: истинные пары подтверждаются высокой долей инлаеров.
4. Итоговые пары прошли проверку платформы, выдав флаг `SIGMOID_VISUAL_ECHO`.

### Ключевые строки исходного кода решения:
```python
# Из solve.py: поиск дубликатов по перцептивным хешам и SIFT+RANSAC
candidates = pHash_cluster(image_hashes, threshold=10)
for p1, p2 in candidates:
    inliers = verify_sift_ransac(p1, p2, ratio_test=0.75)
    if inliers >= 15:
        twin_pairs.append((p1, p2))
# FLAG: SIGMOID_VISUAL_ECHO
```

---

## 4. 🏆 Результат и верификация
- **Итоговый флаг:** `SIGMOID_VISUAL_ECHO`
- **Результат проверки платформы:** Статус: passed, 100/100 баллов.

---

## 5. 📂 Файлы решения в этой папке
- [`analyze.py`](./analyze.py)
- [`analyze2.py`](./analyze2.py)
- [`compute_hashes.py`](./compute_hashes.py)
- [`dl_parallel.py`](./dl_parallel.py)
- [`duputils.py`](./duputils.py)
- [`eval_sift.py`](./eval_sift.py)
- [`eval_zoom.py`](./eval_zoom.py)
- [`features.py`](./features.py)
- [`full_verify.py`](./full_verify.py)
- [`main.py`](./main.py)
- [`main_fast.py`](./main_fast.py)
- [`main_fast18b.py`](./main_fast18b.py)
- [`main_fast2.py`](./main_fast2.py)
- [`main_h18.py`](./main_h18.py)
- [`main_h20.py`](./main_h20.py)
- [`match_pairs.py`](./match_pairs.py)
- [`mphash.py`](./mphash.py)
- [`ncc_all.py`](./ncc_all.py)
- [`output.csv`](./output.csv)
- [`precompute_sift.py`](./precompute_sift.py)
- [`sift_cache.py`](./sift_cache.py)
- [`solution.md`](./solution.md)
- [`solve.py`](./solve.py)
- [`sweep.py`](./sweep.py)
- [`test_verify.py`](./test_verify.py)
- [`train-output.csv`](./train-output.csv)
- [`tune.py`](./tune.py)
- [`verify.py`](./verify.py)

---

## 🚀 Запуск решения
```bash
python solve.py
```
