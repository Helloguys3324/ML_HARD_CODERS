# TASK REPORT — `<task_id>` / `lvl_XX`

> Final per-task report. Filled by the report scribe from artifacts only; no memory-based claims.
> If a number was not observed, write `NOT MEASURED`, not an estimate.

## 0. Scorecard

| Field | Value |
| :--- | :--- |
| Task / country | `task_...` / ... |
| Slot | `lvl_XX` |
| Track | `nlp / tabular / cv / audio-genai / sec / rl / frontier` |
| Input type | text / file / prompt |
| Max score | ... |
| Evaluated score / penalty / final | ... / ... / ... |
| Final score | ... / ... (or `REJECTED`, `PENDING`, `EXPIRED`) |
| Status | CLOSED / EXPIRED / SKIPPED |
| Submission ID | ... (last counted) |
| Submitted at | `...Z` |
| Time used | ... min of ... min limit |
| Best CV | ... (metric: ...) |
| Public/private LB | ... / ... (if visible) |
| Points per hour | ... |

## 1. Goal

What the task asked for, in 2–4 sentences. Include the exact evaluation target and the
input/output contract.

## 2. Data

| File | Bytes | Rows | Cols | Notes |
| :--- | ---: | ---: | ---: | :--- |
| ... | ... | ... | ... | ... |

- Target / label distribution:
- ID / group / time structure:
- Missingness, duplicates, drift:
- CTF/forensics findings (magic, archives, stego, metadata, Unicode, pickle/safetensors):
- Data issues / blockers:

## 3. Fair-Play Compliance

| Rule | Decision | Evidence |
| :--- | :--- | :--- |
| Pretrained models | allowed / forbidden / not used | quote or `N/A` |
| External datasets | allowed / forbidden / not used | quote |
| External APIs | allowed / forbidden / not used | quote |
| Internet at inference | ... | ... |
| Team data sharing | none | — |

One-line compliance statement: ...

## 4. Approach

### 4.1 Iterations

| Run | Model / method | Features / key idea | CV | LB / eval | Time | Decision |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- |
| `r01_baseline` | ... | ... | ... | ... | ... | kept / dropped |
| `r02_...` | ... | ... | ... | ... | ... | kept / dropped |
| `r03_...` | ... | ... | ... | ... | ... | final |

### 4.2 Final Model

- Architecture / algorithm:
- Features / preprocessing:
- CV scheme and folds:
- Seeds / ensemble weights:
- Calibration / threshold:
- Why this candidate was chosen:
- Why alternatives were rejected:

## 5. Validation

- Reproduction command:
- OOF score (metric): ...
- Holdout / LB score: ...
- Sanity checks:
  - [ ] Row count matches sample
  - [ ] Column names/order match
  - [ ] IDs match / order preserved
  - [ ] No NaN / Inf / null / empty strings
  - [ ] Not constant
  - [ ] Values in valid range / class set
  - [ ] No train-test leakage / overlap
  - [ ] Metric direction correct (not inverted)
- Reviewer verdict: `PASS / FAIL`
- Reviewer notes / residual risks:

## 6. Submission Record

| Attempt | File | SHA256 (12) | Bytes | API ID | Status | Score | Time |
| ---: | :--- | :--- | ---: | :--- | :--- | ---: | :--- |
| 1 | `...` | ... | ... | ... | passed/rejected/pending | ... | ...Z |
| 2 | `...` | ... | ... | ... | ... | ... | ... |

- Final counted submission: attempt #..., hash `...`
- Evaluator feedback / details: ...
- Submitted text / prompt (redacted if secret): ...
- Backup path: `work/tasks/<id>/submissions/best_<score>_<sha8>.<ext>`
- Checklist completed: `YES / NO` (link `reports/levels/<id>_SUBMISSIONS.md`)

## 7. Result

- Final score / max: ... / ...
- Passed / rejected / pending: ...
- What the score implies:
- Leaderboard position (if visible):
- Penalties (hints, wrong submissions): ...

## 8. What Worked / What Failed

**Worked:**
- ...

**Failed / wasted time:**
- ...

**Time sinks and how to avoid them next task:**
- ...

## 9. Lessons and Reusable Assets

- Reusable code / config: `work/tasks/<id>/runs/<run>/...`
- Template improvement to backport (if any):
- Playbook update (if any):
- Warning for similar tasks:
- One-line lesson:

## 10. Reproduction

```bash
cd /home/kozak/mllm/MML
# exact command, including env vars and seed
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 POLARS_MAX_THREADS=2 \
/home/kozak/mllm/py lvl_XX/solve.py <args>
```

- Runtime: ... min on 4 CPU / 15 GB
- Determinism: seed ...; known nondeterminism:
- Required files: ...

## 11. Open Items

- [ ] Clue/passphrase collected (if offered): ...
- [ ] Pending evaluation resolved: ...
- [ ] Report committed by orchestrator: ...
- [ ] Backup verified: ...
- [ ] Next task prep: ...

**Scribe:** ... · **Completed at:** `...Z`
