# ЮАР — Ngiyakuzwa? (`task_south_african_republic`)

| Параметр | Значение |
|---|---|
| **Страна / Регион** | **ЮАР (ZA)** |
| **Категория** | **Аудио / Распознавание речи и Шумоподавление** |
| **Сложность** | **Easy** |
| **Баллы** | **100 / 100** (100% максимальный результат) |
| **Флаг решения** | `SIGMOID_ULWAZI` |
| **Статус на платформе** | ✅ **PASSED** |

---

## 1. 📖 Постановка задачи
10-минутная голосовая заметка на языке зулу (isiZulu), записанная на стадионе во время дерби. На фоне шума толпы женщина произносит слово 'ikhodi' ('код'). Слово сразу после него является ключом.

---

## 2. 🔍 В чем состоял подвох
Шум стадиона (вувузелы, крики) полностью глушит речь. Стандартный VAD Whisper отсекает запись как сплошной шум.

---

## 3. ⚙️ Как работает решение (подтверждение строчками кода)
Ниже представлены ключевые строки исходного кода решения из файла `solve.py`, подтверждающие описанный алгоритм:

```python

Run with /home/kozak/mllm/py solve.py
Outputs spectrograms and prints the forensic numbers used in solution.md.
import os
import struct
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
WAV = os.path.join(HERE, "..", "..", "tasks", "task_south_african_republic",
                   "data", "voice_note.wav")
WAV = os.path.abspath(WAV)
OUT = HERE


def wav_chunks(path):
    b = open(path, "rb").read()
    print("file bytes:", len(b), "header:", b[:12])
    pos = 12
    while pos + 8 <= len(b):
        cid = b[pos:pos + 4]
```

---

## 4. 🏆 Результат и верификация
- **Итоговый флаг:** `SIGMOID_ULWAZI`
- **Результат проверки платформы:** Статус: passed, 100/100 баллов.

---

## 5. 📂 Файлы решения в этой папке
- [`answer.txt`](./answer.txt)
- [`denoise.py`](./denoise.py)
- [`preprocess.py`](./preprocess.py)
- [`solution.md`](./solution.md)
- [`solve.py`](./solve.py)
- [`transcribe.py`](./transcribe.py)

---

## 🚀 Запуск решения
```bash
python solve.py
```
