# 08 — Auto-Baseline Generator (first submission in ~10 minutes)

_Agent I, 2026-11-14. Offline, CPU-only, no pretrained weights, no external APIs._

## Files

| File | Purpose |
| :--- | :--- |
| `tools/make_baseline.py` | Detect an unknown task directory and write a valid first submission + `RESULT.json`. |
| `tools/validate_submission.py` | Check a submission against `sample_submission.csv` / task dir before upload. |

## Why

The competition gives us unknown task directories. The first goal is always a
**valid, non-empty, correctly-shaped submission** before spending time on a
better model. This tool buys that baseline in 2–5 minutes on 4 CPU cores.

## Usage

```bash
cd /home/kozak/mllm/MML

# Inspect what the tool detects without writing anything
/home/kozak/mllm/py tools/make_baseline.py --task-dir work/task --dry-run

# Produce a baseline
/home/kozak/mllm/py tools/make_baseline.py --task-dir work/task \
    --output work/task/submissions/submission.csv --time-budget 300 --json

# Validate before submitting
/home/kozak/mllm/py tools/validate_submission.py \
    --submission work/task/submissions/submission.csv --task-dir work/task
```

## Detection logic

1. Recursively discovers CSV/TSV/Parquet/JSON/JSONL tables, image folders,
   audio folders and standalone text files.
2. Finds the train/test split (separate `train`/`test` files, a `split` column,
   or a single table with a holdout) and the `sample_submission.csv`.
3. Infers the ID column, target column, task type (binary / multiclass /
   regression), text columns, image/audio paths and multi-target layouts.
4. Trains the cheapest sensible model:
   - **tabular** → LightGBM (small trees, 1 thread by default on this box),
   - **text** → TF-IDF + linear model,
   - **images** → image statistics + LightGBM (no CNN training),
   - **audio** → MFCC/spectral statistics + LightGBM,
   - **fallback** → target mean/mode.
5. Aligns predictions to `sample_submission.csv` columns/order, scrubs
   NaN/Inf, handles constant columns and unseen categories.
6. Writes `RESULT.json` with CV score, metric, model, runtime and feature count.

### CPU quirk discovered

On this NixOS/OpenMP build, LightGBM multi-threaded histogram construction was
measured **much slower** than single-threaded for small data (12.5 s vs 0.53 s
for 2400×13, 100 trees). The baseline therefore pins LightGBM to one thread by
default; override with `LMML_LGB_JOBS=4` if that changes.

## Validation

19 synthetic task layouts were generated under `work/baseline_test/`
(binary/multiclass/regression tabular, text, tiny images, audio, Parquet with
no sample, image folders, missing targets, JSONL, datetime, single column,
constant target, multiclass-probability layout, multi-target, single-table
split, large tabular/text). **19/19 produced submissions that passed
`validate_submission.py`.**

| Case | Model | CV metric | Wall time |
| :--- | :--- | :--- | ---: |
| binary tabular (2.4k×6) | LightGBM | ROC-AUC 0.983 | 3.4 s |
| multiclass tabular | LightGBM | logloss 0.084 | 2.7 s |
| regression tabular | LightGBM | RMSE 19.22 | 2.3 s |
| text | TF-IDF + linear | AUC 1.000 (synthetic) | 2.0 s |
| tiny images | image stats + LGBM | logloss 0.102 | 3.9 s |
| audio | MFCC stats + LGBM | logloss 0.001 | 3.7 s |
| large tabular | LightGBM | AUC 0.969 | 23.0 s |

Full data: `work/baseline_test/benchmark.json`.

## Limitations / escalation

- This is a **baseline**, not a medal solution. After it passes the validator,
  escalate to the track runbook (`reports/runbooks/`) and the matching template.
- Image/audio baselines use hand-crafted statistics, not CNNs/ASR; fine for a
  first submission, weak for hard tasks.
- Multi-table relational joins and exotic formats (HDF5, SQLite, video) are
  best-effort; inspect `--dry-run` output and fall back manually.
- Always run `validate_submission.py` and the official pre-submit checklist
  before uploading. Remember: **the latest submission counts**, so do not
  replace a good submission with a worse experiment.
