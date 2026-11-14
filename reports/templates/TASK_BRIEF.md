# TASK BRIEF — `<task_id>` / `lvl_XX`

> One page. Filled by intake at T+3, approved by orchestrator by T+5.
> If a field is unknown, write `UNKNOWN` and an owner — do not guess.
> This brief is the solver's only required reading.

## 1. Snapshot

| Field | Value |
| :--- | :--- |
| Task ID | `task_...` |
| Slot / country | `lvl_XX` / ... |
| Title | ... |
| Track / template | `nlp` / `templates/01_nlp_fast.py` |
| Input type | `text flag` / `file` / `prompt` |
| Max score / complexity | ... / ... |
| Task time limit | ... min (server `deadlineAt` = `...Z`) |
| Opened at / hard stop | `...Z` / `...Z` (deadline − max(5 min, 10%)) |
| Orchestrator / shift | ... |

## 2. Mission

One sentence: what must be produced and submitted.

> ...

## 3. Rules Extract (quote the task where possible)

| Rule | Value | Source / note |
| :--- | :--- | :--- |
| Primary metric / direction | ... / `max` or `min` | ... |
| ID column / target column | ... / ... | ... |
| Submission shape | rows × cols, columns, order | ... |
| Accepted file types / max size | ... / ... MB | task UI/API |
| Flag format (if text) | `...` | ... |
| Pretrained models | **allowed / forbidden / unspecified** | quote: "..." |
| External data | **allowed / forbidden / unspecified** | quote: "..." |
| External API / internet | **allowed / forbidden / unspecified** | quote: "..." |
| Hints / penalty | ... points | ... |
| Submission limit / latest counts | ... | platform rule |
| Other restrictions | ... | ... |

**Fair-play decision (default deny):** use `from-scratch / provided data / no external API`
unless a row above says allowed. Record the exact enabling quote.

## 4. Data Manifest

| File | Bytes | SHA256 (8) | Rows | Cols | Notes |
| :--- | ---: | :--- | ---: | ---: | :--- |
| `files/train.csv` | ... | ... | ... | ... | ... |
| `files/test.csv` | ... | ... | ... | ... | ... |
| `files/sample_submission.csv` | ... | ... | ... | ... | ... |

- Target distribution / class balance:
- For non-CSV data: file inventory + count / dimensions / duration:
- Missingness / duplicates / ID overlap:
- Sample submission parsed: `YES / NO` — mismatches:
- Missing or blocked files:

## 5. Recon Findings (T+5–T+15)

- **EDA:** ...
- **CTF / forensics:** magic bytes, archives, stego, metadata, Unicode, pickle/safetensors, canary:
- **Leakage / drift:** ID/time leak, train-test overlap, adversarial validation:
- **Baseline hypothesis:** ...

## 6. Plan

| Phase | Owner | Command / approach | ETA | Output |
| :--- | :--- | :--- | :--- | :--- |
| Baseline | ... | ... | ... | `runs/r01_baseline/` |
| Improve 1 | ... | ... | ... | `runs/r02_.../` |
| Improve 2 (optional) | ... | ... | ... | `runs/r03_.../` |
| Fallback | ... | constant-but-valid / previous best / flag guess | ... | ... |

**Current best / fallback file + SHA256:** `...` / `...`
**Hints used (penalty):** ...

**Exact baseline command:**

```bash
cd /home/kozak/mllm/MML
# Example for nlp (01) / tabular (02). `solve` runs solve.py with cwd = lvl_XX/,
# so template paths are challenge-relative. Full flag lists: see the cheat sheet
# in reports/11_RUNBOOK_CONSISTENCY.md.
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 POLARS_MAX_THREADS=2 \
/home/kozak/mllm/py lmml_master.py solve lvl_XX -- \
  --train data/train.csv --test data/test.csv --sample-sub data/sample_submission.csv
# Keep the template out-dir at its default (the challenge dir) so solve.py writes
# lvl_XX/RESULT.json and lvl_XX/submissions/submission.csv, which lmml_master
# reads to update the vault. Copy them to work/tasks/<task_id>/runs/r01_.../ after.
```

Template-specific first arguments (replace the `--train/--test/--sample-sub`
group; `--out-dir` is accepted by all seven):

| Track | Key args |
| :--- | :--- |
| `nlp` (01) | `--mode tfidf --text-col ... --target-col ... --no-pretrained` |
| `tabular` (02) | `--target ... --models lgb --folds 3 --no-use-gpu` |
| `cv` (03) | `--mode train --data-dir data --img-size 96 --epochs 2 --no-pretrained` |
| `audio`/`genai` (04) | `--mode audio_clf --data-dir data --arch conformer --no-pretrained` |
| `sec` (05) | `--mode forensics --input data` (path is challenge-relative) |
| `rl` (06) | `--env CartPole-v1 --algo ppo --steps 20000 --device cpu` |
| `frontier` (07) | `--mode pinn --pinn-pde poisson1d --device cpu` |

## 7. Validation Plan

- CV scheme: `stratified 5-fold` / `group` / `time-series` / `holdout` — why:
- Primary metric and direction:
- Sanity checks: row count, columns, IDs, NaN/Inf, constant, range, class mapping
- Reviewer assigned:
- Leakage audit focus:

## 8. Risks / Unknowns / Escalation Triggers

| Risk | Likelihood | Impact | Trigger | Fallback |
| :--- | :--- | :--- | :--- | :--- |
| ... | ... | ... | ... | ... |

## 9. Clock

| Milestone | UTC | Status |
| :--- | :--- | :--- |
| T+0 intake | ... | ... |
| T+5 EDA | ... | ... |
| T+15 baseline start | ... | ... |
| T+35 improvement start (baseline submitted) | ... | ... |
| T+55 validation / ensemble | ... | ... |
| T+70 improved submit | ... | ... |
| T+80 report v1 | ... | ... |
| T+80–100 reserve wave (optional) | ... | ... |
| T+100–105 final submit / confirm | ... | ... |
| T+108 hard stop (10% margin) | ... | — |
| T+108–120 evaluation buffer | ... | — |

## 10. Artifacts

- Brief: `reports/levels/<id>_BRIEF.md`
- Manifest: `work/tasks/<id>/manifest.json`
- Runs: `work/tasks/<id>/runs/`
- Validation: `reports/levels/<id>_VALIDATION.md`
- Submission ledger: `work/tasks/<id>/submissions/ledger.jsonl`
- Final report: `reports/levels/<id>.md`

**Approved by orchestrator:** `YES / NO` — time: `...Z`
