# SUBMISSION CHECKLIST — `<task_id>` / `<candidate>`

> One checklist per candidate. **Any unchecked box = do not submit.**
> Mark `N/A` only on items explicitly labelled *(if applicable)*; never skip a core check.
> Submission officer runs it; orchestrator gives the final GO.
> Latest submission per task counts, so a bad upload can replace a good one.

## 0. Authority and Timing

- [ ] Reviewer verdict is `PASS` (`reports/levels/<id>_VALIDATION.md`).
- [ ] Orchestrator GO recorded (time + initials).
- [ ] No teammate has a `pending` / evaluating submission for this task (team-wide lock).
- [ ] Deadline margin: at least 5 min (or 10% of remaining, whichever is larger) before the server deadline.
- [ ] Candidate is better than the current counted submission, **or** it is the known-good fallback.
- [ ] Current best file + hash are backed up and can be re-submitted if this attempt regresses.

## 1. Candidate Identity

- [ ] Task ID is exactly `<task_id>` (no country-code alias mismatch).
- [ ] Run ID / config / seed recorded.
- [ ] File hash recorded:

```bash
sha256sum <candidate_file>
stat -c '%n %s bytes' <candidate_file>
file <candidate_file>
```

- [ ] Candidate path is the frozen one (not a file still being written).
- [ ] Candidate was produced by the command in `RESULT.json` / `solve.py`.

## 2. File Type, Size, and Shape

- [ ] Extension is in the task's accepted list (`.csv`, `.zip`, `.json`, `.txt`, ...).
- [ ] MIME/content type matches the extension; no HTML error page saved as `.csv`.
- [ ] File size ≤ task `maxUploadedFileSize`.
- [ ] File is non-empty and not truncated.
- [ ] Archive *(if applicable)* extracts cleanly and contains the required files:

```bash
unzip -t <file>.zip        # or: tar -tf <file>.tar.gz
```

- [ ] Row count matches `sample_submission` (CSV) or the task's required count *(if applicable)*.
- [ ] No extra index column, no extra header row, no footer.
- [ ] Encoding is UTF-8 (or the task's required encoding); line endings are consistent.

## 3. Schema (CSV / table submissions)

- [ ] Column names match `sample_submission` **exactly** (case, spaces, punctuation).
- [ ] Column order matches the sample (if the evaluator is order-sensitive).
- [ ] ID column name and dtype match; leading zeros preserved (read IDs as strings).
- [ ] ID values match the sample's set and order.
- [ ] Target column(s) present and correctly named.
- [ ] No duplicate IDs; no missing IDs.
- [ ] For multi-class: class labels match the allowed set/order; no unknown labels.

Automated gate (preferred validator first; `lmml_master.py check` is the repo gate):

```bash
cd /home/kozak/mllm/MML
/home/kozak/mllm/py tools/validate_submission.py \
  --submission work/tasks/<task_id>/submissions/<candidate>.csv \
  --sample work/tasks/<task_id>/files/sample_submission.csv --json
/home/kozak/mllm/py lmml_master.py check lvl_XX \
  --sub work/tasks/<task_id>/submissions/<candidate>.csv \
  --sample work/tasks/<task_id>/files/sample_submission.csv
```

Strict check (works even without `lmml_master.py`):

```bash
/home/kozak/mllm/py - <<'PY'
import polars as pl, numpy as np
sub = pl.read_csv("<candidate>.csv")
sam = pl.read_csv("<sample_submission>.csv")
assert sub.height == sam.height, (sub.height, sam.height)
assert sub.columns == sam.columns, (sub.columns, sam.columns)
assert sum(sub.null_count().row(0)) == 0, "nulls"
for c, dt in zip(sub.columns, sub.dtypes):
    if dt in (pl.Float32, pl.Float64):
        a = sub[c].to_numpy()
        assert np.isfinite(a).all(), f"NaN/Inf in {c}"
        assert np.std(a) > 0, f"constant {c}"
assert (sub[sam.columns[0]].cast(pl.Utf8) == sam[sam.columns[0]].cast(pl.Utf8)).all(), "ID mismatch"
print("PASS", sub.shape)
PY
```

Flag-only check:

```bash
/home/kozak/mllm/py - <<'PY'
import re
flag = open("<flag_file>").read().strip()
assert re.fullmatch(r"LMML\{[A-Za-z0-9_\-!@#$%^&*+=.:/?]+\}", flag), repr(flag)
print("PASS", flag)
PY
```

(Use the task's actual flag pattern if different; do not assume `LMML{}`.)

## 4. Values

- [ ] No `NaN`, `Inf`, `-Inf`, `null`, empty string, or whitespace-only cell.
- [ ] Probabilities in `[0, 1]`; logits/probabilities not mixed *(if applicable)*.
- [ ] Regression predictions within the plausible target range; no absurd outliers *(if applicable)*.
- [ ] Classification labels in the allowed set; no float labels when strings are expected *(if applicable)*.
- [ ] Not constant (unless the task explicitly allows it and the orchestrator approved).
- [ ] Text/flag *(if applicable)*: exact format, no trailing newline, no Markdown fences, no quotes.
- [ ] Prompt *(if applicable)*: one clear answer, no secrets/tokens, no instructions that violate task rules.

## 5. Semantics

- [ ] Metric direction is correct (`max` vs `min`); predictions are not inverted.
- [ ] Threshold/calibration applied exactly as validated; class mapping consistent.
- [ ] Row order matches the sample (or IDs are explicitly used by the evaluator).
- [ ] No train/test contamination: test IDs were not used for fitting; no target leakage.
- [ ] If CV is suspiciously high (> 0.99), reviewer has explicitly cleared it as genuine *(N/A if not applicable)*.
- [ ] Baseline sanity: a trivial/constant prediction would score worse; candidate is not accidentally empty.
- [ ] For time-series: no future information; split is causal *(if applicable)*.

## 6. Task Rules and Fair Play

- [ ] Pretrained models: used only if the task explicitly allows them; quote recorded.
- [ ] External datasets: used only if explicitly allowed; source recorded.
- [ ] External APIs: used only if explicitly allowed; no hidden calls at inference.
- [ ] No internet access during evaluation (unless the task requires it).
- [ ] No hard-coded test labels, evaluator probing beyond normal submissions, or leakage from other teams.
- [ ] File naming and packaging match the task instructions.
- [ ] No secrets, tokens, personal data, or other teams' data in the submission.
- [ ] Submission does not contain executable code unless the task asks for it.

## 7. Backup and Reproducibility

- [ ] Backup copy made **before** upload:

```bash
mkdir -p work/tasks/<task_id>/submissions
cp <candidate_file> work/tasks/<task_id>/submissions/best_<score>_<sha8>.<ext>
```

- [ ] Hash of the backup matches the candidate.
- [ ] `RESULT.json` updated with `submission`, `sha256`, `cv_score`, metric, config.
- [ ] Run command + seed saved in `run.log` / `config.json`.
- [ ] A second copy exists outside the repo or in `work/` (gitignored).
- [ ] If the file is > 50 MB: it will **not** be committed; only the hash + pointer are recorded.

## 8. API Execution

- [ ] Token valid (`tools/sigmoid_api.py me`), submissions open (`settings.allowSubmissions`).
- [ ] Correct task ID; correct submission type (`text`, `file`, or `prompt`).
- [ ] File flow: `POST /submission/upload-file-url` → `PUT` presigned URL → `POST /submission`.
- [ ] Upload response `allowed=true`, `key` present; PUT returned 2xx.
- [ ] POST response contains a submission id/status; no `submissionPending` / `submissionThrottled` error.
- [ ] Request/response recorded in `ledger.jsonl` (no token in logs).

```bash
cd /home/kozak/mllm/MML
/home/kozak/mllm/py tools/sigmoid_api.py submit-file \
  --task-id <task_id> --file <candidate_file>
# or
/home/kozak/mllm/py tools/sigmoid_api.py submit-text \
  --task-id <task_id> --text '<flag>'
```

For prompt tasks, use `POST /submission/text` with the prompt as `text` (max 15,000 characters); each prompt is evaluated independently and only the latest counts.

## 9. Post-Submit Verification

- [ ] Poll status until `passed` / `rejected` / `pending` (back off 10/30/60 s).
- [ ] Record API id, status, score, max score, feedback, timestamp.
- [ ] `reports/levels/<id>_SUBMISSIONS.md` updated.
- [ ] `reports/LIVE_STATUS.md` / task board updated by the proper owner (not hand-edited if watcher owns it).
- [ ] If `passed`: update best score + backup name.
- [ ] If `rejected`: record reason; decide whether to re-submit the previous best.
- [ ] If `pending`: do not submit again; wait; use the time for the report.
- [ ] If error: record exact body; escalate per the matrix; do not blind-retry.

## 10. Rollback

- [ ] Previous best file still exists and its hash matches the ledger.
- [ ] If this candidate regressed: re-submit the previous best (only if time and no pending evaluation).
- [ ] If the evaluator is stuck: notify the human captain; keep the best file backed up.
- [ ] If the deadline passed: stop; record `EXPIRED`; the last accepted submission stands.

## Sign-Off

| Role | Name / session | Verdict | Time (UTC) |
| :--- | :--- | :--- | :--- |
| Reviewer | ... | PASS / FAIL | ... |
| Submission officer | ... | READY / BLOCKED | ... |
| Orchestrator | ... | GO / NO-GO | ... |

**Final decision:** `SUBMIT` / `HOLD` / `ROLLBACK`
**Reason (one line):** ...
