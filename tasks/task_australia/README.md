# Австралия — The Drunken Mario Adventures (`task_australia`)

| Параметр | Значение |
|---|---|
| **Страна / Регион** | **Австралия (AU)** |
| **Категория** | **Реверс-инжиниринг / Анализ политик обучения с подкреплением** |
| **Сложность** | **Medium** |
| **Баллы** | **200 / 200** (100% максимальный результат) |
| **Флаг решения** | `SIGMOID_TANH` |
| **Статус на платформе** | ✅ **PASSED** |

---

## 1. 📖 Постановка задачи
Приложение Marathon Lab (автономный бинарник Linux x86_64) моделирует эволюцию популяции агентов на уровне платформера. Обученные агенты приобретают странное поведение, в котором зашифровано кодовое слово.

---

## 2. 🔍 В чем состоял подвох
Бинарник скомпилирован через PyInstaller и запускается в графическом режиме, скрывая внутреннюю логику.

---

## 3. ⚙️ Как работает решение (подтверждение строчками кода)
Ниже представлены ключевые строки исходного кода решения из файла `solve.py`, подтверждающие описанный алгоритм:

```python

What the app is
---------------
Marathon Lab is a Nuitka one-file Pygame application.  The one-file payload
(``KA`` + zstd stream at the end of the executable, terminated by an 8-byte
little-endian payload length) contains a standalone distribution with
``marathon-lab.bin``, ``lab/proven.npy`` and the Pygame runtime.

The relevant game data (recovered by importing the compiled modules inside the
running app; see ``probe_runtime.py``) is:

  lab.level.COIN_SITES = [7, 12, 17, 25, ... 206]   (20 coins, all on row 11)
  lab.fitness._SIGN    = {(7, 11): 1, (12, 11): 0, ...}

``lab.fitness.reward`` pays +COIN_WEIGHT (25) for coins whose sign is 1 and
-COIN_WEIGHT for coins whose sign is 0.  A merely competent player would grab
every coin.  A fully trained agent learns to collect exactly the +1 coins and
to leave the -1 coins alone.

``lab/proven.npy`` is the bundled, fully-trained reference brain.  Running it
is deterministic: it reaches the flagpole in 1197 steps (fitness 514.21,
``won=True``) and takes exactly the seven +1 coins:
    7, 17, 80, 101, 110, 118, 172
and none of the thirteen -1 coins.  (Verified 3/3 identical runs.)

```

---

## 4. 🏆 Результат и верификация
- **Итоговый флаг:** `SIGMOID_TANH`
- **Результат проверки платформы:** Статус: passed, 200/200 баллов.

---

## 5. 📂 Файлы решения в этой папке
- [`answer.txt`](./answer.txt)
- [`coin_signs.csv`](./coin_signs.csv)
- [`evidence_module_dump.txt`](./evidence_module_dump.txt)
- [`evidence_reward_events.txt`](./evidence_reward_events.txt)
- [`evidence_runtime.txt`](./evidence_runtime.txt)
- [`probe_reward_events.py`](./probe_reward_events.py)
- [`probe_runtime.py`](./probe_runtime.py)
- [`pyinstxtractor.py`](./pyinstxtractor.py)
- [`pyinxtractor-ng.py`](./pyinxtractor-ng.py)
- [`run_fhs.sh`](./run_fhs.sh)
- [`sitecustomize.py`](./sitecustomize.py)
- [`solution.md`](./solution.md)
- [`solve.py`](./solve.py)
- [`writers.py`](./writers.py)

---

## 🚀 Запуск решения
```bash
python solve.py
```
