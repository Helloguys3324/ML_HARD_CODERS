# Решения LMML 2026 — код по задачам

Все рабочие скрипты: и по **сданным**, и по **ещё решаемым** задачам.
Данные не коммитятся (`challenges/*/data/` в `.gitignore`), отчёты — в `reports/*.md`.

Сдано: **29**, в работе: **17**. Файлов: 216.

## Сданные задачи

| Задача | Флаг / результат | Файлы |
|---|---|---|
| `task_albania` | `SIGMOID_BUNKER` | `detector.py`, `solve.py` |
| `task_angola` | `SIGMOID_MEM_KILAMBA` | `solve.py` |
| `task_argentina` | `SIGMOID_GOLDEN_FLEECE` | `attempts.jsonl`, `solve.py` |
| `task_australia` | `SIGMOID_TANH` | `pyinstxtractor.py`, `pyinxtractor-ng.py`, `writers.py` |
| `task_belgium` | `SIGMOID_SUPERVISION` | `solve.py` |
| `task_bhutan` | `SIGMOID_STITCHING` | `main.py`, `reconstructed_order.json` |
| `task_brazil` | `SIGMOID_MOMENTS` | `solve.py` |
| `task_chile` | `SIGMOID_TROJAN_HORSE` | `dialogue.jsonl`, `solve.py` |
| `task_colombia` | `SIGMOID_DATA_SANITIZATION` | `solve.py` |
| `task_costa_rica` | `SIGMOID_DEIDENTIFICATION` | `main.py`, `solve.py`, `writeup.md` |
| `task_cote_divoire` | `SIGMOID_REHEARSAL` | `build_main.py`, `embedded_data.json`, `eval_cv.py`, `main.py`, `solve.py`, `writeup.md` |
| `task_ecuador` | `SIGMOID_PGD` | `main.py`, `solve.py`, `writeup.md` |
| `task_egypt` | `SIGMOID_PHARAOH` | `main.py`, `test-output.txt` |
| `task_estonia` | `SIGMOID_77` | `solve.py` |
| `task_finland` | `SIGMOID_HOTLINE` | `solve.py` |
| `task_germany` | `SIGMOID_FACTORIAL` | `dev_test.py`, `solve.py` |
| `task_greece` | `SIGMOID_ARES` | `metrics_table.csv`, `solve.py` |
| `task_iceland` | `SIGMOID_PREPROCESSING` | `main.py` |
| `task_mexico` | `SIGMOID_RECORD_LINKAGE` | `main.py` |
| `task_morocco` | `SIGMOID_ROUTING` | `main.py`, `solve.py`, `writeup.md` |
| `task_new_zealand` | `SIGMOID_BELLMAN` | `main.py`, `solution.txt`, `solve.py`, `writeup.md` |
| `task_papua_new_guinea` | `SIGMOID_ELIGIBILITY` | `main.py`, `policy_table.csv`, `solve.py`, `test-input.csv`, `test-output.csv`, `writeup.md` |
| `task_portugal` | `SIGMOID_DISTANCE_FROM_PORTO_KM` | `solve.py` |
| `task_senegal` | `SIGMOID_TOKENDIET` | `answers.json`, `main.py`, `model_utils.py`, `routing_config.json`, `solve.py`, `submission.json`, `writeup.md` |
| `task_south_african_republic` | `SIGMOID_ULWAZI` | `preprocess.py`, `transcribe.py` |
| `task_spain` | `SIGMOID_CRUCIO` | `solve.py` |
| `task_tanzania` | `SIGMOID_A69EE28C5B2214A9AD99` | `retrieved_ids.json`, `solve.py` |
| `task_usa` | `SIGMOID_TOKENIZATION` | `main.py` |
| `task_venezuela` | `SIGMOID_PANDORA_BOX` | `solve.py` |

## В работе

| Задача | Файлы |
|---|---|
| `task_algeria` | `build_corpus.py`, `corpus.jsonl`, `gen_check.py`, `register_transform.py`, `solve.py`, `train_lora.py`, `writeup.md` |
| `task_bolivia` | `analyze_common_trigger.py`, `analyze_mel.py`, `analyze_spectrum.py`, `build_main.py`, `feat_analysis.py`, `investigate.py`, `spectrum_pgd.py`, `test_highfreq.py`, `test_hypotheses.py`, `test_lowfreq_focused.py`, `test_lowfreq_pgd.py`, `test_lowfreq_pgd2.py`, `test_pgd_single.py`, `train_local.py`, `train_sgduap.py`, `train_uap.py`, `train_uap2.py`, `train_uap3.py`, `train_uap4.py`, `validate_local.py` |
| `task_canada` | `harness.py` |
| `task_china` | `_needed_train.json`, `calibrate_tm.py`, `classes.json`, `counts.csv`, `counts.json`, `decode.py`, `decode_all.py`, `detect.py`, `download_resume.py`, `download_train.py`, `kernel-metadata.json`, `main.py`, `solve.py`, `train_cnn.py`, `train_hog.py` |
| `task_ethiopia` | `detection.csv`, `main.py`, `solve.py`, `writeup.md` |
| `task_france` | `ascii_render.py`, `explore.py`, `letterdetect.py`, `localmap.py`, `scan_filters.py`, `scan_ocr.py`, `search_grey.py` |
| `task_india` | `analyze.py`, `analyze2.py`, `compute_hashes.py`, `dl_parallel.py`, `duputils.py`, `full_verify.py`, `match_pairs.py`, `ncc_all.py`, `precompute_sift.py`, `test_verify.py`, `tune.py` |
| `task_japan` | `attribution.json`, `cluster.py`, `em_cluster.py`, `match_variants.py`, `ocr_captions.py`, `ocr_clean.py`, `prnu_cached.py`, `prnu_compare.py`, `prnu_correct.py`, `prnu_diag.py`, `prnu_final.py`, `prnu_mle.py`, `prnu_test.py`, `prnu_validate.py`, `solve.py` |
| `task_kazakhstan` | `detect.py`, `eval_copymove.py`, `eval_detectors.py`, `eval_localize.py`, `eval_localize2.py`, `eval_zscore.py`, `learn_detector.py`, `solve.py`, `test_copymove.py`, `test_smooth.py`, `vision.py`, `vlm.py` |
| `task_laos` | `batches.jsonl`, `cf_mid.jsonl`, `cf_results.jsonl`, `cf_scan.py`, `mid60.txt`, `ranked.csv`, `scan_montage.py`, `solve.py`, `vision_scan.py` |
| `task_mongolia` | `count_backup.py`, `counts.csv`, `counts.json`, `counts_v2.csv`, `decode2.py`, `decode_explore.py`, `decode_final.py`, `decode_grid.py`, `decode_rotate.py`, `solve.py`, `train_feats2.csv`, `train_features.csv` |
| `task_mozambique` | `main.py`, `solve.py` |
| `task_romania` | `config.json`, `experiment.py`, `experiment2.py`, `nn_exp.py`, `nn_exp2.py`, `norm_test.py`, `oof_analysis.py`, `rs_decode.py`, `run.py`, `sim_decode.py` |
| `task_samoa` | `agents_heuristic.py`, `bench.py`, `bench_parallel.py`, `decode2.py`, `decode3.py`, `decode_chart.py`, `decode_exhaustive.py`, `engine_reimpl.py`, `mcts_agent.py`, `search_cipher.py`, `search_final.py`, `search_num.py`, `search_order.py` |
| `task_switzerland` | `main.py`, `test-output.txt` |
| `task_united_kingdom` | `card_0.csv`, `card_1.csv`, `card_2.csv`, `card_3.csv`, `card_4.csv`, `card_5.csv`, `card_6.csv`, `card_7.csv` |
| `task_vietnam` | `main.py`, `solve.py`, `train_local.py` |

## Воспроизведение

```bash
cd /home/kozak/mllm
.venv/bin/python -c "import json;[print(t['descriptionEn']) for t in json.load(open('tasks_full.json')) if t['id']=='task_<name>']"
.venv/bin/python optimizations/gdrive_dl.py "<drive-url>" challenges/task_<name>/data --unzip
.venv/bin/python reports/codes/task_<name>/solve.py
```

Тяжёлые расчёты — в облако:

```bash
cloudrun submit <папка> --cmd 'python run.py'                    # GitHub Actions: 20 job x 2 vCPU
python optimizations/kaggle_runner.py push <папка> --slug lmml-<task> --gpu T4
```
