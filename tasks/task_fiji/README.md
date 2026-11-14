# Фиджи — Walu Season (`task_fiji`)

| Параметр | Значение |
|---|---|
| **Страна / Регион** | **Фиджи (FJ)** |
| **Категория** | **Обучение с подкреплением / Multi-Armed Bandits** |
| **Сложность** | **Easy** |
| **Баллы** | **100 / 100** (100% максимальный результат) |
| **Флаг решения** | `SIGMOID_BAGGING` |
| **Статус на платформе** | ✅ **PASSED** |

---

## 1. 📖 Постановка задачи
В прибрежных водах рифа Кадаву есть 10 рыболовных мест и 1000 ежедневных забросов на протяжении 100 дней. Обычные места дают стабильно 1-2 кг рыбы, но в некоторых местах периодически проходят косяки валу (улов до 20-25 кг). Требуется алгоритм, максимизирующий суммарный сезонный улов (порог прохождения >= 5300 кг/день).

---

## 2. 🔍 В чем состоял подвох
Стандартный эпсилон-жадный алгоритм или наивный UCB1 либо застревают в исследовании низкодоходных точек, либо слишком поздно находят редкие всплески валу.

---

## 3. ⚙️ Как работает решение (подтверждение строчками кода)
Ниже представлены ключевые строки исходного кода решения из файла `main.py`, подтверждающие описанный алгоритм:

```python

Strategy
--------
Each day 10 spots are shown in a fresh random order, so everything must be
learned from scratch within the day.  The catch distributions have a very
characteristic shape: most spots are "steady" (a kilo or two, low variance)
while a few are "school" spots (usually nothing, but now and then ~20 kg).
Neither plain greedy nor plain UCB1 handles that well:

* greedy locks onto the first lucky spot and never leaves it;
* UCB1 with a raw sample mean is fooled by a lucky 20 kg catch on a spot that
  almost never produces one.

The Strategy below therefore uses a small empirical-Bayes mixture model per
spot plus a UCB exploration index:

1.  All catches from all spots are pooled and split at an adaptive threshold
    into a "big" cluster (walu schools, ~20 kg) and a "small" cluster
    (everything else).  The threshold is re-estimated from the pooled data at a
    few checkpoints, so it scales automatically with the reef.
2.  For each spot we keep the counts and sums of big/small catches and place a
    Beta(1, 1) posterior on the probability ``p`` that a cast hits a school,
    and normal posteriors on the two component means.  The posterior mean of a
    spot is ``p * big_mean + (1 - p) * small_mean`` and its posterior variance
    follows from the delta method.  The Beta prior is what prevents both
```

---

## 4. 🏆 Результат и верификация
- **Итоговый флаг:** `SIGMOID_BAGGING`
- **Результат проверки платформы:** Статус: passed, 100/100 баллов, средний улов 6728 кг/день.

---

## 5. 📂 Файлы решения в этой папке
- [`bench.py`](./bench.py)
- [`bench2.py`](./bench2.py)
- [`candidates.py`](./candidates.py)
- [`evaluation_script.py`](./evaluation_script.py)
- [`final_validate.py`](./final_validate.py)
- [`main.py`](./main.py)
- [`prior_test.py`](./prior_test.py)
- [`reef.py`](./reef.py)
- [`robust.py`](./robust.py)
- [`robust2.py`](./robust2.py)
- [`solution.md`](./solution.md)
- [`train-input.json`](./train-input.json)
- [`train-output.json`](./train-output.json)
- [`tune.py`](./tune.py)

---

## 🚀 Запуск решения
```bash
python evaluation_script.py main.py
```
