# Отчет о решении: task_usa ("The Great American Token Mystery")

## 1. Обзор и цель
- **Task ID:** `task_usa` (США).
- **Сложность:** Medium / Reverse Engineering.
- **Тип отправки решения:** Исполняемый скрипт `main.py`, оцениваемый оффлайн в изолированной среде проверки.
- **Требования и контракт:**
  - Считывает `test-input.json`, содержащий:
    ```json
    {
      "calibration": {"text": "...", "token_ids": [...]},
      "challenge":   {"token_ids": [...]}
    }
    ```
  - «Вслепую» определяет, какой подсловный токенизатор (subword tokenizer) сгенерировал `calibration.token_ids` из `calibration.text`.
  - Декодирует `challenge.token_ids`, используя определенный токенизатор.
  - Записывает `test-output.json`:
    ```json
    {
      "tokenizer_id": "<id>",
      "decoded_message": "<text>"
    }
    ```
- **Ограничения платформы:**
  - Строго оффлайн-проверка (без доступа к интернету; внешние библиотеки, такие как `tiktoken`, `transformers`, `tokenizers` и `sentencepiece`, недоступны).
  - Максимальный размер файла: $\le 10.0$ МБ. Время выполнения: менее 10 секунд.
- **Цель:** Создать автономный `main.py` на чистой стандартной библиотеке Python (stdlib), содержащий встроенные сжатые словари для всех основных семейств токенизаторов NLP, способный выполнять детектирование и точную реконструкцию строк без внешних зависимостей.

---

## 2. Ключевые наблюдения и методология

### 2.1 Калибровка на обучающей выборке (`train-input.json`)
- На калибровочном сплите обучающих данных идентификаторы токенов идеально совпали с **`p50k_base`** (OpenAI Davinci / Code-Davinci) с точностью 73/73 токена.
- Декодирование `challenge.token_ids` через `p50k_base` в точности воспроизвело эталонное сообщение из `train-output.json`:
  `"TRAIN_SAMPLE: Verification successful at the American NLP Observatory. Token IDs map to p50k_base vocabulary!"`.

### 2.2 Архитектура встроенного реестра (Embedded Registry)
- Чтобы обойти ограничения на внешние библиотеки, словари для **22 уникальных токенизаторов** были сериализованы в единую бинарную нагрузку через `lzma` + `base85` + `pickle` и внедрены прямо в код `main.py`.
- Итоговый размер скрипта: **8.67 МБ** (с запасом укладывается в лимит 10.0 МБ).
- **Поддерживаемые семейства (22 словаря):**
  - **tiktoken:** `gpt2` (`r50k_base`), `p50k_base`, `cl100k_base` (GPT-4), `o200k_base` (GPT-4o).
  - **HuggingFace Byte-level BPE:** `roberta-base`/`bart`, `Llama-3-8B` (128k), `Qwen2.5-0.5B` (151k), `SmolLM2-135M`, `Falcon-7B`, `StarCoder2-3B`, `CodeGen-350M`/`phi-2`.
  - **HuggingFace WordPiece:** `bert-base-uncased`, `bert-base-cased`, `bert-base-multilingual-cased`.
  - **SentencePiece / Metaspace:** `t5-small`, `albert-base-v2`, `xlnet-base-cased`, `Llama-2-7B`/`TinyLlama`, `Mistral-7B-v0.1`, `Phi-3-mini`, `DeBERTa-v3-small`, `XLM-RoBERTa-base`.

### 2.3 Движок токенизации на чистом Python и идентификация
- **Пользовательские декодеры:** Реализации на чистом Python, обрабатывающие байтовые токены (byte fallback), сборку байтов UTF-8, метасимвол пробела SentencePiece ` `, отображение символов byte-level BPE и склейку префиксов `##` в WordPiece.
- **Пайплайн скоринга:** Декодирует `calibration.token_ids` по словарям-кандидатам и оценивает сходство с `calibration.text`, используя:
  1. Точное побайтовое совпадение с нормализацией пробелов и регистра.
  2. Удаление специальных токенов (`[CLS]`, `<s>`, `<pad>`, `[SEP]`).
  3. Сходство строк через `difflib.SequenceMatcher`.
  4. Жесткие штрафы за токены вне словаря (Out-Of-Vocabulary / OOV), предотвращающие доминирование надмножеств словарей над подмножествами (например, при различии `gpt2` и `p50k_base`).
- **Самопроверка (Self-Verification):** Извлекает имя кандидата, заложенное в шаблоне сообщения (`"Token IDs map to <name> vocabulary!"`), для кросс-валидации ранжирования кандидатов.

### 2.4 Оценка на тестовой выборке (`test-output.json`)
- При проверке на скрытом тестовом наборе алгоритм однозначно определил токенизатор **`cl100k_base`**.
- Декодированная полезная нагрузка раскрыла флаг задания:
  ```
  SIGMOID_TOKENIZATION
  Congratulations! You have successfully solved the American Tokenizer Mystery.
      Verification Hash: 2026-USA-NLP-BPE
          Cluster Node: Grand-Canyon-Compute-01
  Subword token IDs are arbitrary vocabulary matrix indices, not universal ASCII codes!
  ```

---

## 3. Детали реализации

1. **`main.py` (Скрипт для отправки, 8.67 МБ):**
   - Распаковывает встроенные словари:
     ```python
     raw_data = pickle.loads(lzma.decompress(base64.b85decode(VOCAB_DATA_B85)))
     ```
   - Содержит легковесные декодеры: `TiktokenDecoder`, `ByteBpeDecoder`, `SentencePieceDecoder`, `WordPieceDecoder`.
   - Производительность: общее время выполнения ~3.5–4.5 с, пиковое потребление памяти ~155 МБ.

2. **`build_registry.py` и `build_main.py` (Сборщик артефактов):**
   - Считывает исходные `tokenizer.json` из HuggingFace, файлы SentencePiece `.model` и дампы tiktoken, компилируя отображения токен-байт в `main_template.py`.

3. **`validate.py` (Проверка паритета):**
   - Сравнивает декодеры на чистом Python с нативными библиотеками `tiktoken`, `tokenizers` и `sentencepiece` на 11 сложных краевых последовательностях (эмодзи, многоязычный текст, апострофы и нестандартные пробелы).
   - Получено **0 расхождений по всем 22 токенизаторам**.

4. **`synthetic_test.py` (Сквозной стресс-тест):**
   - Генерирует 22 синтетических задания по каждому поддерживаемому токенизатору.
   - Результат: **22/22 тестов успешно пройдены** с точной идентификацией `tokenizer_id` и 100% восстановлением символов.

---

## 4. Результаты и артефакты
- **Артефакт отправки:** `main.py` (чистая стандартная библиотека Python, 8.67 МБ).
- **Определенный тестовый токенизатор:** `cl100k_base`
- **Извлеченный флаг:** `SIGMOID_TOKENIZATION` (проверочный хеш: `2026-USA-NLP-BPE`).
- **Ключевые артефакты (`task_usa/`):**
  - `main.py`: Автономный скрипт решения со встроенными упакованными словарями.
  - `test-output.json`: Проверенный тестовый вывод (`tokenizer_id: cl100k_base`, сообщение с флагом).
  - `answer.txt`: Сводка верификации.
  - `answer_guess.json`: Предсказание локальной валидации.
  - `build_registry.py` и `build_main.py`: Скрипты бинарной компиляции и сборки пакета.
  - `validate.py`: Скрипт аудита соответствия официальным библиотекам токенизации.
  - `synthetic_test.py`: Стенд тестирования на 22 синтетических бенчмарках.
  - `solution.md`: Полная документация по обратному инжинирингу.
