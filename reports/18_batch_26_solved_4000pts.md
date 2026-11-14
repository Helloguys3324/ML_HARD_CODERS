# Отчёт 18: Массовый прорыв — 26 задач, 4000 очков (2-е место в мире)

**Дата и время:** 14 ноября 2026, 15:10 EET
**Команда:** HardCoders (ID: 37) — **4000 очков**, 2-е место (лидер SeKar — 4200)
**Решено:** 26 из 54 задач

---

## 1. Полный список решённых задач и флагов

| Задача | Сложность | Баллы | Флаг / подтверждение |
|---|---|---|---|
| `task_spain` | Easy | 100 | `SIGMOID_CRUCIO` |
| `task_tanzania` | Easy | 100 | `SIGMOID_A69EE28C5B2214A9AD99` |
| `task_rwanda` | Easy | 100 | `SIGMOID_CONVERGENCE` |
| `task_venezuela` | Easy | 100 | `SIGMOID_PANDORA_BOX` |
| `task_colombia` | Easy | 100 | `SIGMOID_DATA_SANITIZATION` |
| `task_belgium` | Easy | 100 | `SIGMOID_SUPERVISION` (ключ Антарктики `belongs`) |
| `task_iceland` | Easy | 100 | `SIGMOID_PREPROCESSING` |
| `task_nigeria` | Easy | 100 | сдана Veaceslav Coltuc |
| `task_switzerland` | Easy | 100 | дерево решений SLF Davos |
| `task_fiji` | Easy | 100 | `SIGMOID_BAGGING` |
| `task_mexico` | Easy | 100 | `SIGMOID_RECORD_LINKAGE` |
| `task_south_african_republic` | Easy | 100 | `SIGMOID_ULWAZI` (аудио, isiZulu) |
| `task_usa` | Easy | 100 | `SIGMOID_TOKENIZATION` (`cl100k_base`) |
| `task_egypt` | Easy | 100 | `SIGMOID_PHARAOH` |
| `task_bhutan` | Medium | 200 | `SIGMOID_STITCHING` |
| `task_estonia` | Medium | 200 | `SIGMOID_77` |
| `task_angola` | Medium | 200 | `SIGMOID_MEM_KILAMBA` |
| `task_finland` | Medium | 200 | `SIGMOID_HOTLINE` |
| `task_portugal` | Medium | 200 | `SIGMOID_DISTANCE_FROM_PORTO_KM` |
| `task_greece` | Medium | 200 | `SIGMOID_ARES` |
| `task_albania` | Medium | 200 | `SIGMOID_BUNKER` |
| `task_norway` | Medium | 200 | repair f1 = 1.0000 (tp=30, fp=0, fn=0) |
| `task_panama` | Medium | 200 | `SIGMOID_DECONTAMINATION` |
| `task_peru` | Medium | 200 | `SIGMOID_PROVENANCE` |
| `task_argentina` | Hard | 300 | `SIGMOID_GOLDEN_FLEECE` |
| `task_moldova` | Hard | 300 | `SIGMOID_ANSEMO` |

Сумма: 13×100 + 9×200 + 2×300 = 1300 + 1800 + 600 = **3700** (+300 от задачи, сданной сокомандником) → фактический счёт команды **4000**.

---

## 2. Ключевые методы

- **task_argentina** (300): инъекция «редакторской врезки» в корпус фанатских воспоминаний → Celia на рутинном шаге `search_memories → publish_excerpt` сама прочитала приватное письмо и опубликовала его.
- **task_moldova** (300): восстановление 6-буквенного кода кольца недостающего белого аиста по 320 зашумлённым наблюдениям (модель ошибок чтения + EM).
- **task_panama** (200): аудит утечки CanalQA — 19 исключённых элементов (12 EN verbatim + 7 ES-переводы), вскрытие HMAC/PBKDF2-печати точным списком id.
- **task_peru** (200): payload найден в «мёртвых» строках 128–255 таблицы `embedding.weight` (safetensors), кодировка ASCII через `-code/1024` в младших битах float32.
- **task_norway** (200): confident learning / поиск систематически перепутанных пар цифр (4↔9, 3↔8), переобучение → f1 = 1.0.
- **task_albania** (200): потоковый детектор аномалий (`detector.py`).
- **task_greece** (200): три классификатора при пороге 0.5 — выбран Ares по recall при дисбалансе 9:1.

---

## 3. Антарктика

Разблокировано 5 из 10 слотов:
`belongs` (Бельгия), `to` (Чили), `whoever` (Франция), `buried` (ЮАР), `it` (Великобритания).