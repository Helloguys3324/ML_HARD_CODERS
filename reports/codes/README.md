# Решения LMML 2026 — код по задачам

Здесь лежат рабочие скрипты по каждой сданной задаче (то, чем реально получен флаг).
Полные отчёты — в `reports/*.md`, данные не коммитятся (`challenges/*/data/` в `.gitignore`).

## Индекс

### Easy (100)

| Задача | Флаг / результат | Файлы |
|---|---|---|
| `task_spain` | `SIGMOID_CRUCIO` | `solve.py` |
| `task_tanzania` | `SIGMOID_A69EE28C5B2214A9AD99` | `solve.py`, `retrieved_ids.json` |
| `task_rwanda` | `SIGMOID_CONVERGENCE` | `graph.py` (исправленный `inspect_batch`) |
| `task_venezuela` | `SIGMOID_PANDORA_BOX` | `solve.py` |
| `task_colombia` | `SIGMOID_DATA_SANITIZATION` | `solve.py` |
| `task_belgium` | `SIGMOID_SUPERVISION` | `solve.py` |
| `task_iceland` | `SIGMOID_PREPROCESSING` | `main.py` |
| `task_switzerland` | дерево решений SLF (проверка: `test-output.txt matches`) | `main.py` |
| `task_fiji` | `SIGMOID_BAGGING` | `main.py` (UCB1-tuned), `reef.py` |
| `task_mexico` | `SIGMOID_RECORD_LINKAGE` | `solve.py` |
| `task_south_african_republic` | `SIGMOID_ULWAZI` | `solve.py`, `transcribe.py` |
| `task_usa` | `SIGMOID_TOKENIZATION` | `main.py` |
| `task_egypt` | `SIGMOID_PHARAOH` | `main.py` |
| `task_nigeria` | `SIGMOID_TALKINGDRUM` | код не сохранился (сдавал сокомандник) |

### Medium (200)

| Задача | Флаг / результат | Файлы |
|---|---|---|
| `task_bhutan` | `SIGMOID_STITCHING` (SSIM 1.0) | `main.py` |
| `task_estonia` | `SIGMOID_77` | `solve.py` |
| `task_angola` | `SIGMOID_MEM_KILAMBA` | `solve.py` |
| `task_finland` | `SIGMOID_HOTLINE` | `solve.py` |
| `task_portugal` | `SIGMOID_DISTANCE_FROM_PORTO_KM` | `solve.py` |
| `task_greece` | `SIGMOID_ARES` | `solve.py`, `metrics_table.csv` |
| `task_albania` | `SIGMOID_BUNKER` | `detector.py`, `solve.py` |
| `task_norway` | repair f1 = 1.0 (tp=30, fp=0, fn=0) | код не сохранился |
| `task_panama` | `SIGMOID_DECONTAMINATION` | `analyze.py`, `report.json` |
| `task_peru` | `SIGMOID_PROVENANCE` | код не сохранился (метод — в отчёте) |
| `task_chile` | `SIGMOID_TROJAN_HORSE` | `solve.py`, `dialogue.jsonl` |
| `task_senegal` | `SIGMOID_TOKENDIET` | `solve.py`, `routing_config.json`, `main.py` |
| `task_cote_divoire` | `SIGMOID_REHEARSAL` | `solve.py`, `build_main.py`, `main.py`, `eval_cv.py` |
| `task_costa_rica` | `SIGMOID_DEIDENTIFICATION` (privacy/utility 1.0) | `main.py`, `solve.py`, `writeup.md` |
| `task_papua_new_guinea` | `SIGMOID_ELIGIBILITY` (2.33×) | `solve.py`, `main.py`, `policy_table.csv` |
| `task_morocco` | `SIGMOID_ROUTING` (q 0.852, cost 9980) | `solve.py`, `main.py` |
| `task_australia` | `SIGMOID_TANH` | `pyinstxtractor.py`, `run_stdout.log` |
| `task_ecuador` | `SIGMOID_PGD` (conf 0.9992) | `main.py`, `solve.py`, `experiment*.py` |
| `task_laos` | 20/28 дефектных ракушек (0.7143) | `solve.py`, `cf_scan.py`, `ranked.csv` |

### Hard (300)

| Задача | Флаг / результат | Файлы |
|---|---|---|
| `task_argentina` | `SIGMOID_GOLDEN_FLEECE` | `solve.py`, `attempts.jsonl` |
| `task_moldova` | `SIGMOID_ANSEMO` | код не сохранился (метод — в отчёте) |
| `task_new_zealand` | `SIGMOID_BELLMAN` (293 мин) | `solve.py`, `main.py` |
| `task_germany` | `SIGMOID_FACTORIAL` (140/150 запросов) | `solve.py` |
| `task_brazil` | backdoor `yolo11n-cls`, все 6 гейтов | `solve.py` (`model.pt` не коммитится — вес > лимита) |

## Как воспроизводить

```bash
cd /home/kozak/mllm
.venv/bin/python -c "import json;[print(t['descriptionEn']) for t in json.load(open('tasks_full.json')) if t['id']=='task_<name>']"
.venv/bin/python optimizations/gdrive_dl.py "<drive-url>" challenges/task_<name>/data --unzip
.venv/bin/python reports/codes/task_<name>/solve.py
```

Тяжёлые расчёты — в облако, не локально:

```bash
cloudrun submit <папка> --cmd 'python run.py'     # GitHub Actions: 20 job × 2 vCPU
python optimizations/kaggle_runner.py push <папка> --slug lmml-<task> --gpu T4
```