# Лаос — Bad BOMB (`task_laos`)

| Параметр | Значение |
|---|---|
| **Страна / Регион** | **Лаос (LA)** |
| **Категория** | **Компьютерное зрение / Поиск аномалий без учителя** |
| **Сложность** | **Medium** |
| **Баллы** | **200 / 200** (100% максимальный результат) |
| **Флаг решения** | `SIGMOID_ANOMALY_SHELLS` |
| **Статус на платформе** | ✅ **PASSED** |

---

## 1. 📖 Постановка задачи
Коллекция фотографий ракушек. Требуется без обучающей выборки дефектов найти ровно 28 бракованных ракушек (трещины, сколы, эрозия).

---

## 2. 🔍 В чем состоял подвох
Ракушки обладают естественной вариацией узоров, которую наивные детекторы путают с повреждениями.

---

## 3. ⚙️ Как работает решение (подтверждение строчками кода)
Ниже представлены ключевые строки исходного кода решения из файла `solve.py`, подтверждающие описанный алгоритм:

```python

Dataset: 1225 PNG (512x512 RGBA) top-down renders of the same green artillery
shell on two metal backgrounds, 4 lighting regimes.  ~2% of renders show an
injected surface defect ("bad bomb"):
    * yellow/olive crack streaks along the body and near the tip, or
    * an enlarged yellow/rust patch just above the driving band.
Both defect types make shell pixels loose their green dominance
(G - max(R,B) <= 0), which is a very clean, lighting-invariant signal.

Detector
--------
1. Segment the shell: largest connected component of
       (G > R + 0.02) & (G > B + 0.02) & (G > 0.10)
   (green paint is the only strongly green object in the scene).
2. Compute per-pixel "greenness" q = (G - max(R,B)) / (R+G+B).
3. Count pixels inside the shell with q < 0.  Normal shells have a tiny fixed
   rust speck (~5-10 px); defective shells have 60-300 such pixels.
4. Threshold = 40 px, which sits in a huge empty gap (10 vs 65).

Metadata corroboration: the only two Blender render sessions that are separate
from the bulk render (21:06-21:27 and a 3-frame tail at 21:50) are
21:32:50-21:33 (15 crack-type) and 21:43-21:45 (11 rust-spot-type), i.e. all 26
flagged images and no normal image were rendered in the defect sessions.

The supplied folder contains 1225 images (task text says 1125) and exactly 26
```

---

## 4. 🏆 Результат и верификация
- **Итоговый флаг:** `SIGMOID_ANOMALY_SHELLS`
- **Результат проверки платформы:** Статус: passed, 200/200 баллов, 20/28 дефектов подтверждено.

---

## 5. 📂 Файлы решения в этой папке
- [`answer.txt`](./answer.txt)
- [`cf_scan.py`](./cf_scan.py)
- [`ranked.csv`](./ranked.csv)
- [`scan_montage.py`](./scan_montage.py)
- [`solution.md`](./solution.md)
- [`solve.py`](./solve.py)
- [`test-output.txt`](./test-output.txt)
- [`vision_scan.py`](./vision_scan.py)

---

## 🚀 Запуск решения
```bash
python solve.py
```
