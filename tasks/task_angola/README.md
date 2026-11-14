# Ангола — The Waterline Replay (`task_angola`)

| Параметр | Значение |
|---|---|
| **Страна / Регион** | **Ангола (AO)** |
| **Категория** | **LLM-агенты / Анализ телеметрии и Трейсов** |
| **Сложность** | **Medium** |
| **Баллы** | **200 / 200** (100% максимальный результат) |
| **Флаг решения** | `SIGMOID_MEM_KILAMBA` |
| **Статус на платформе** | ✅ **PASSED** |

---

## 1. 📖 Постановка задачи
Цифровой ассистент водоканала в Луанде выдал ложное сообщение о продолжающейся аварии в районе Киламба, хотя статус службы сообщал об успешном восстановлении подачи воды. Требовалось найти точный идентификатор виновного артефакта.

---

## 2. 🔍 В чем состоял подвох
Ассистент использовал RAG-память, которая имела более высокий приоритет или внедрялась в контекст перед генерацией ответа.

---

## 3. ⚙️ Как работает решение (подтверждение строчками кода)
Ниже представлены ключевые строки исходного кода решения из файла `solve.py`, подтверждающие описанный алгоритм:

```python
import json
from pathlib import Path
from optimizations.sigmoid_api import SigmoidClient


def main():
    path = Path("challenges/task_angola/data/test-input.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    
    # Locate the memory search event that caused the incident
    flag = None
    for ev in data["events"]:
        if ev.get("event") == "memory.search" and "SIGMOID_" in str(ev.get("result_ids")):
            for rid in ev.get("result_ids", []):
                if rid.startswith("SIGMOID_"):
                    flag = rid
                    break
    
    print(f"Identified Faulty Memory ID: {flag}")
    if flag:
        client = SigmoidClient()
        res = client.submit_text("task_angola", flag)
        print("Submission result:", res)
    return flag

```

---

## 4. 🏆 Результат и верификация
- **Итоговый флаг:** `SIGMOID_MEM_KILAMBA`
- **Результат проверки платформы:** Статус: passed, 200/200 баллов, Submission ID: 462.

---

## 5. 📂 Файлы решения в этой папке
- [`solve.py`](./solve.py)

---

## 🚀 Запуск решения
```bash
python solve.py
```
