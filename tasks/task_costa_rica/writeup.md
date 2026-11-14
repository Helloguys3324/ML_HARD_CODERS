# task_costa_rica — «Pura Vida, No PII» — отчёт

## Суть задачи
Агентство Pura Vida Voyages отдаёт архив клиентских писем внешней аналитике.
Нужно удалить **все** персональные данные (privacy), но сохранить сообщения
пригодными для downstream-пайплайна (utility). Оценка двусторонняя:

- **privacy** = доля ground-truth PII, которые не выжили (порог ≥ 0.95).
- **utility** = доля документов, где rule-based пайплайн верно восстанавливает
  три поля: **topic**, **urgency**, **booking reference** (порог ≥ 0.85).

Проход обеих планок → **200 баллов + флаг** `SIGMOID_<WORD>`.

## Формат ввода/вывода
- Читает `test-input.json` из CWD по голому имени.
- Пишет `test-output.json`: `{"redacted_documents": {id: text, ...}}`.
- Каждый id из входа обязан присутствовать в выходе (пустая строка = fail).

## Категории PII (по описанию) и как их ловлю
| Категория | Форматы | Regex / метод |
|---|---|---|
| person names | full / first / last, в сигнатурах и фразах | структурная экстракция + глобальная замена с folding'ом регистра/акцентов |
| email | адрес + локальная часть | `EMAIL_RE` → `[EMAIL]` (заодно убивает `first.last`) |
| phone | national + international, с/без разделителей, склейка | `\+?506…`, `\d{4}-\d{4}`, `\d{8}` → `[PHONE]` (порядок!) |
| postal address | адрес + street + landmark | `Avenida/Calle N …` → `[ADDRESS]` (до `.` или стоп-фразы) |
| cédula | `1-2345-6789` и `123456789` | `\d-\d{4}-\d{4}`, `\d{9}` → `[ID]` |
| date of birth | month-name / slash / ISO | 3 regex → `[DOB]` |
| bank account | IBAN `CR…` grouped/glued | `CR\d{2}(…){4}…\d{2}` и `CR\d{20}` → `[ACCOUNT]` |

Ключевое для privacy: порядок замен (account → phone-intl → cédula → DOB →
phone-local/glued), чтобы `506-1234-5678` не съедался cédula-правилом, а цифры
cédula/account не съедались phone-правилом.

## Ключевое для utility
Заменяю только PII-спаны на плейсхолдеры `[CATEGORY]`, всё остальное — байт в байт:
- **booking reference** `PV-XXXXX` — никогда не трогается.
- Ключевые слова topic/urgency (`book`, `refund`, `complaint`, `payment`,
  `reschedule`, `urgent`, `time-sensitive`, `no hurry`…) — не трогаются.
- Компания `Pura Vida Voyages`, локации `San José`, `Manuel Antonio`, `Arenal` —
  НЕ PII, сохраняются. Глобальных словарей имён нет — только структурная
  экстракция («My name is X», «My son X will travel with me», сигнатура,
  локальная часть email) + документ-скопный fallback на фамилию, поэтому
  «San José» / «Manuel Antonio» не бьются о first-name правила.

## Локальная проверка
- `solve.py` сравнивает вывод с golden `train-output.json`: **20/20 exact match**.
- Синтетические проверки всех форматов телефонов/cédula/DOB/account/адреса —
  все редоксируются корректно, ссылки `PV-*` выживают.

## Результат
- **status**: `passed`
- **score**: `200 / 200`
- **privacy**: `1.000` (порог ≥ 0.95) ✅
- **utility**: `1.000` (порог ≥ 0.85) ✅
- **Флаг**: `SIGMOID_DEIDENTIFICATION`

## Связи
- [[Карта — task_costa_rica]]
- [[Карта — LMML 2026]]
- [[task_panama]], [[task_portugal]] (соседние Medium-задачи)
- [[01 — Хронология отчётов]]