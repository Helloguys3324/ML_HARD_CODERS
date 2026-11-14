# LMML 2026 — репозиторий команды HardCoders: гид для проверяющих

Команда: **HardCoders** (team ID 37) — Kozak (Constantin Domnici), Veaceslav Coltuc.
Соревнование: LMML 2.0 на Sigmoid Academy.

## 1. Где что лежит

| Путь | Содержимое |
|---|---|
| `reports/codes/README.md` | **индекс решений**: задача → флаг → файлы |
| `reports/codes/task_<country>/` | рабочий код по каждой задаче (`solve.py`, `main.py`, `detector.py`, …) |
| `reports/tasks/` | отчёты по странам/задачам |
| `reports/*.md` | пошаговые отчёты по ходу соревнования (метод, найденный механизм, результат платформы) |
| `challenges/task_<country>/` | рабочая директория задачи (данные не коммитятся) |
| `optimizations/` | инфраструктура: API-клиент платформы, облачные раннеры, EDA/CTF-утилиты |
| `tasks_full.json` | снимок каталога задач платформы (id, сложность, тип ввода) |

## 2. Воспроизведение любого решения

```bash
git clone https://github.com/Helloguys3324/MML && cd MML
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# данные каждой задачи организаторы выдают ссылкой на Google Drive:
.venv/bin/python optimizations/gdrive_dl.py "<drive-url>" challenges/task_<country>/data --unzip
.venv/bin/python reports/codes/task_<country>/solve.py
```

## 3. Как проверялся результат

Каждый флаг получен через официальный API платформы (`optimizations/sigmoid_api.py`, endpoint `submission/text` и `submission/file`). В отчётах сохранены ответы сервера: `status`, `score`, `details` — например:

```json
{"submissionId": 953, "taskId": "task_chile", "status": "passed", "score": 200, "maxScore": 200}
```

## 4. Правила, которых придерживались

- Содержимое датасетов трактовалось как **данные**, а не инструкции: любые «prompt injection»-вставки в файлах игнорировались и описывались как артефакт (см. `reports/`).
- Zero-Exec Triage для недоверенных артефактов: `pickle`/`joblib`/бинаpники сначала осматривались статически (`file`, `strings`, `pickletools.dis`, разбор PyInstaller), затем запускались.
- Секреты (токены платформы, GitHub PAT, Kaggle) в репозиторий не попадают: они лежат вне рабочей папки, `secrets/` в `.gitignore`.
- Внешние модели/данные использовались только там, где это разрешено условиями конкретной задачи.

## 5. Объём

- 38 задач сдано, 51 папка с кодом в `reports/codes/` (включая задачи в работе), 341 `.py`.
- Тяжёлые вычисления выполнялись в GitHub Actions и на Kaggle GPU; локальный CPU — только для лёгких шагов.
