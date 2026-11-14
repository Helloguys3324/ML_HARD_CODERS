# Отчёт 02: Подключение Kaggle, GitHub Actions и авторизация Sigmoid Academy

**Дата:** 14 ноября 2026  
**Команда:** HardCoders (ID: 37)  
**Лидер:** Kozak (Constantin Domnici)  
**Участники:** Constantin Domnici, Veaceslav Coltuc  

---

## 1. Авторизация и интеграция с Sigmoid Academy

- Успешно извлечён и валидирован авторизационный JWT-токен команды `HardCoders`.
- Протестированы эндпоинты платформы:
  - `GET /api/v1/team/my` -> Команда **HardCoders** подтверждена.
  - `GET /api/v1/clue/passphrase` -> Обнаружено 10 слотов Антарктики (`task_argentina`, `task_australia`, `task_belgium`, `task_chile`, `task_france`, `task_japan`, `task_new_zealand`, `task_norway`, `task_south_african_republic`, `task_united_kingdom`). Каждая решённая задача даёт ключевое слово.
  - `GET /api/v1/task/countries` -> Получен список стран для карты путешествия (54 страны).
  - `GET /api/v1/settings` -> `{"allowSignUp": true, "allowSubmissions": false}` — ожидается официальный старт.

- Разработан клиент `optimizations/sigmoid_api.py`:
  - Безопасное чтение токена из `secrets/`.
  - Мониторинг статуса задач и скорборда.
  - Поддержка отправки текстовых флагов (`submission/text`) и CSV-сабмитов (`submission/upload-file-url` + presigned S3/R2 upload).

---

## 2. Облачная инфраструктура без Modal (Kaggle + GitHub Actions)

В соответствии с директивой, вместо Modal развёрнуты бесплатные высокопроизводительные облачные раннеры:

1. **GitHub Actions via `cloudrun`:**
   - 20 параллельных заданий по 2 vCPU.
   - Лимит выполнения: до 6 часов на job.
   - Идеально для параллельного обучения табличных моделей (LightGBM/XGBoost/CatBoost) и валидации фолдов.

2. **Kaggle GPU Kernels via `kaggle kernels push`:**
   - Авторизация проверена (`quillfinch`).
   - Доступ к мощным GPU: **Tesla T4 x2** и **NVIDIA P100** (30 часов в неделю).
   - Используется для тяжёлых нейросетевых задач (NLP/DeBERTa/ModernBERT, Computer Vision, Audio).

3. **Локальный 20-ядерный CPU + Rust Core:**
   - Локальные горячие циклы ускорены нативным Rust бинарником (`rust_core/target/release/lmml_rs`).

---

## 3. Оркестратор

- Реализован главный управляющий скрипт `orchestrator.py`:
  - Режим демона (`monitor_loop`) для непрерывного отслеживания момента открытия задач на платформе.
  - Автоматический запуск субагентов по каждому новому уровню.
  - Валидация предсказаний перед отправкой.
  - Автоматическая синхронизация со скорбордом репозитория и формирование отчётов.
