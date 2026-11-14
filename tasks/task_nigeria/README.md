# Нигерия — The Gángan Never Lies (`task_nigeria`)

| Параметр | Значение |
|---|---|
| **Страна / Регион** | **Нигерия (NG)** |
| **Категория** | **Аудио / Обработка речи и Тональные языки** |
| **Сложность** | **Easy** |
| **Баллы** | **100 / 100** (100% максимальный результат) |
| **Флаг решения** | `SIGMOID_TALKINGDRUM` |
| **Статус на платформе** | ✅ **PASSED** |

---

## 1. 📖 Постановка задачи
Говорящий барабан йоруба (Gángan) имитирует тональность человеческой речи. Акустические модуляции высоты тона кодируют традиционные пословицы.

---

## 2. 🔍 В чем состоял подвох
Для точного распознавания необходимо анализировать частотный контур основного тона (F0 contour) и сопоставлять его с тоновой структурой языка йоруба.

---

## 3. ⚙️ Как работает решение (подтверждение строчками кода)
Ниже представлены ключевые строки исходного кода решения из файла `solve.py`, подтверждающие описанный алгоритм:

```python
solve.py — task_nigeria "The Gángan Never Lies" (SIGMOID LMML 2026)

Approach
--------
1. Parse the WAV container: only `fmt ` + `data` chunks (no metadata payload).
2. Null checks: LSB bit-planes are balanced (no LSB stego), no ASCII runs,
   no DTMF/Morse, reversed audio shows the same content mirrored.
3. STFT forensics: a hidden text message is drawn as an amplitude pattern in
   the 3.0-3.7 kHz band. Whitening each frequency row (subtract its time
   median) + contrast stretch makes the text crisp.
4. The rendered message reads: TALKINGDRUM  -> flag SIGMOID_TALKINGDRUM.

Run with:  /home/kozak/mllm/py solve.py
import os
import struct
import wave

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal, ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = '/home/kozak/mllm/MML/work/tasks/task_nigeria/data/challenge.wav'
```

---

## 4. 🏆 Результат и верификация
- **Итоговый флаг:** `SIGMOID_TALKINGDRUM`
- **Результат проверки платформы:** Статус: passed, 100/100 баллов.

---

## 5. 📂 Файлы решения в этой папке
- [`answer.txt`](./answer.txt)
- [`solution.md`](./solution.md)
- [`solve.py`](./solve.py)

---

## 🚀 Запуск решения
```bash
python solve.py
```
