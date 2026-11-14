# 12 — Submission Officer (safe submission wrapper)

_Agent K, 2026-11-14 09:20–09:30 UTC. No live submissions were made._

## 0. TL;DR

`tools/submit_officer.py` turns a submission into a deliberate, auditable
operation: **claim → preflight → dry-run → `--yes` → backup → API → audit**.
It never reads, prints or copies the JWT; HTTP is delegated to
`tools/sigmoid_api.py`. It cannot silently replace a good submission: every run
prints the latest-submission warning, and a previously passed task needs an
explicit `--allow-resubmit`.

Dry-run tests against the currently locked API passed (no network, no crash,
clear refusal for unclaimed tasks). The claim table is empty again and no
`work/state/submissions.jsonl` exists, i.e. nothing was submitted.

---

## 1. Why this tool exists

The platform scores **only the latest submission per task for the whole team**.
A later rejected/worse attempt replaces a passed one; hints deduct points and
are irreversible. The dangerous failure mode is not "no submission" — it is
"someone resubmitted a good task at 03:00 and made it worse".

`submit_officer.py` enforces four rules:

1. **One writer per task** — a claim table with an owner.
2. **No accidental submit** — `--yes` is mandatory for a real request;
   `--dry-run` is the default recommendation.
3. **Cheap mistakes fail locally** — wrong input type, extension, size or bad
   CSV is rejected before the upload URL is requested.
4. **Everything is recoverable/auditable** — exact bytes are backed up and the
   API response is appended to a JSONL log.

---

## 2. Files and state

| Path | Purpose | Git |
|---|---|---|
| `tools/submit_officer.py` | the wrapper (new) | tracked |
| `reports/12_SUBMISSION_OFFICER.md` | this report | tracked |
| `work/state/claims.json` | claim table (`task_id → owner`) | ignored (`work/`) |
| `work/state/claims.lock` | `flock` guard for claim updates | ignored |
| `work/state/submissions/<utc>_<task>_<name>` | byte-exact backup of every real submission | ignored |
| `work/state/submissions.jsonl` | audit: timestamp, owner, hashes, API response | ignored |
| `work/state/submissions.lock` | `flock` guard for audit appends | ignored |

`work/` is already in `.gitignore`; no task data, submission or token can leak
into the repository through this tool. The tool does not open the token file
at all — the child `sigmoid_api.py` process does.

---

## 3. Workflow

```
assign task
   │
   ├─ claim <task> --owner <name>          # one owner per task
   │
   ├─ solve locally
   │
   ├─ submit-file/text ... --dry-run       # validation + exact request preview
   │      └─ reviewer checks output        # optional second pair of eyes
   │
   ├─ submit-file/text ... --yes           # only the claim owner
   │      ├─ backup bytes → work/state/submissions/
   │      ├─ tools/sigmoid_api.py → platform
   │      └─ append response → work/state/submissions.jsonl
   │
   └─ wait for evaluation (pending blocks the whole team)
         ├─ passed → do NOT resubmit; keep the claim until the event ends
         └─ rejected/partial → owner decides on one improved attempt
```

`--dry-run` performs **all** local checks and prints the request; it exits
before the backup/API branch. `--yes` without `--dry-run` is the only way to
send anything.

---

## 4. Commands

All commands use the NixOS wrapper (`/home/kozak/mllm/py`), never bare
`python`.

### 4.1 Claim protocol (2+ teammates)

```bash
cd /home/kozak/mllm/MML
PY=/home/kozak/mllm/py
TOOL=tools/submit_officer.py

# orchestrator assigns a task to one agent
$PY $TOOL claim task_md --owner alice --note "Europe / tabular, ETA 40m"

# anyone can inspect without changing state
$PY $TOOL status
$PY $TOOL status --json

# teammate tries to take the same task -> refused (exit 2)
$PY $TOOL claim task_md --owner bob

# deliberate takeover only (logs previousOwner + forced=true)
$PY $TOOL claim task_md --owner bob --force

# owner releases; release without --owner is refused for safety
$PY $TOOL release task_md --owner alice
$PY $TOOL release task_md --owner bob --force   # emergency only
```

Rules for the team:

- **One claim = one writer.** Only the claim owner runs a real `--yes`.
- Claim before starting work, not after modelling.
- Keep the claim after a pass. It is a "do not touch" marker, not a lock to be
  cleaned up.
- Release only when abandoning the task or after the event; an orchestrator may
  use `--force` with a note.
- The whole team shares one pending slot per task: if evaluation is running,
  **nobody** can submit that task. Check with
  `$PY tools/sigmoid_api.py submissions` before retrying.

### 4.2 File submission

```bash
# 1) dry-run: claim + inputType + extension + size + validator + request preview
$PY $TOOL submit-file \
  --task-id task_md \
  --file work/tasks/task_md/submissions/submission.csv \
  --owner alice --dry-run

# 2) real submission, after reviewing the preview
$PY $TOOL submit-file \
  --task-id task_md \
  --file work/tasks/task_md/submissions/submission.csv \
  --owner alice --yes
```

Optional: `--url https://...`, `--allow-validation-fail` (loud warning),
`--skip-validate` (discouraged), `--max-size-mb N` (only when task metadata is
missing or wrong).

### 4.3 Text / flag / prompt submission

```bash
$PY $TOOL submit-text --task-id task_ar --text 'flag{...}' --owner alice --dry-run
$PY $TOOL submit-text --task-id task_ar --text 'flag{...}' --owner alice --yes

# long prompts: avoid shell quoting problems
$PY $TOOL submit-text --task-id task_xx --text-file work/task_xx/prompt.txt \
  --owner alice --dry-run
```

Prompt tasks are capped at 15,000 characters by the platform; the tool rejects
longer text locally. A plain `text` task has no such cap (only a warning above
15k).

### 4.4 Exit codes

| Code | Meaning |
|---:|---|
| 0 | dry-run OK, or real submission accepted by the API |
| 2 | preflight refusal (claim, input type, size, extension, resubmit guard) |
| 3 | real submission attempted without `--yes`; nothing sent |
| 4 | CSV validator failed; nothing sent |
| 5 | `sigmoid_api.py` returned non-zero (submission may or may not have landed) |

---

## 5. Preflight checks (in order)

1. **Claim** — task must exist in `claims.json` and belong to `--owner`
   (`--no-claim` skips this with a loud warning).
2. **Previous submission guard** — reads the last local audit record:
   - `passed` → real submit refused unless `--allow-resubmit`; dry-run only
     warns.
   - `pending` → real submit refused unless `--allow-pending`; dry-run only
     warns.
3. **Task metadata** — loads `work/tasks/<task_id>/task.json` when present:
   - `inputType=file` required for `submit-file`; `text`/`prompt` allowed for
     `submit-text`; a mismatch is a hard error.
   - `acceptedFileExtensions` (`*`/empty = any) checked against the file name.
   - `maxUploadedFileSize` checked. The value is treated as **MB** when
     `≤ 2048`, as **bytes** otherwise (admin UI range is 0.01–2047.99 MB); if
     metadata is absent the platform default **10 MB** is used.
   - No metadata (normal while `/task` is locked) is a warning, not a failure.
4. **File sanity** — exists, non-empty, has an extension, size within limit.
5. **CSV validator** — runs
   `tools/validate_submission.py --submission <file> --task-dir work/tasks/<task_id> --json`.
   The task dir is always passed explicitly so the validator cannot rglob the
   whole repo and compare against an unrelated sample. If the task dir/sample
   is absent it still runs generic checks (NaN/Inf, duplicates, constants,
   parseability). Validator errors block the submit unless
   `--allow-validation-fail`.
6. **URL field** — optional `--url` must be absolute `http(s)`.
7. **Warning banner** — printed before every run, dry-run included.

Request previews are exact: text shows `POST /submission/text {taskId,text,url?}`;
file shows the three-step presigned flow
(`POST /submission/upload-file-url` → `PUT <presigned>` →
`POST /submission {taskId,fileStorageKey,url?}`). The presigned URL and upload
key are placeholders in the preview because the API has not been called.

---

## 6. Latest-submission safeguards

| Risk | Safeguard |
|---|---|
| Two agents submit the same task | claim table + owner check; duplicate claim refused unless `--force` |
| Accidental placeholder/test submit | `--dry-run` prints and exits; real submit needs `--yes` |
| Replacing a passed task with a worse attempt | local audit status `passed` blocks real submit unless `--allow-resubmit` |
| Submitting while evaluation is pending (team-wide 409) | local status `pending` blocks real submit unless `--allow-pending` |
| Wrong format/type/extension/size | inputType, extension, size and CSV validation fail before network |
| Bad upload cannot be reconstructed | bytes copied to `work/state/submissions/` before the API call |
| "Did it land?" after a timeout | JSONL records API return code/response; tool tells the operator to query `sigmoid_api.py submissions` instead of blind-retrying |
| Token exposure | tool never reads the token; only the `sigmoid_api.py` child does |
| Concurrent claim/audit writes | `fcntl.flock` around claim updates and JSONL appends; atomic JSON replace |

The local `passed`/`pending` guard is only as fresh as `submissions.jsonl`.
After a failed/timed-out call, verify with the platform before retrying; a 409
`submissionPending` means the whole team is locked until evaluation finishes.

---

## 7. Test evidence (2026-11-14, API still locked)

Environment: `/home/kozak/mllm/py`, `GET /settings` →
`{"allowSignUp":true,"allowSubmissions":false}`, `/task` → 403. All dry-runs
were executed with `SIGMOID_API_BASE=http://127.0.0.1:9/` — if the tool had
tried any network call it would have failed; it did not. No token was read or
printed.

### 7.1 Claim table

```text
$ ... claim task_dryrun_test --owner kozak --note "dry-run test only"
claimed: task_dryrun_test -> kozak

$ ... claim task_dryrun_test --owner teammate
REFUSED: task_dryrun_test is already claimed by 'kozak' at 2026-11-14T09:26:21Z.
exit=2

$ ... claim task_dryrun_test --owner teammate --force
FORCED takeover: task_dryrun_test kozak -> teammate

$ ... status
TASK              OWNER  CLAIMED               NOTE  LAST
task_dryrun_test  kozak  2026-11-14T09:26:21Z        -
```

`release` without `--owner`, with the wrong owner, and with the correct owner
returned refusal/refusal/success respectively (exit 2/2/0). The test claims
were released afterwards; `work/state/claims.json` now has `claims: {}`.

### 7.2 Unclaimed task → clear refusal, no crash

```bash
SIGMOID_API_BASE=http://127.0.0.1:9/ \
/home/kozak/mllm/py tools/submit_officer.py submit-text \
  --task-id task_dryrun_unclaimed --text 'flag{test}' --owner kozak --dry-run
```

```text
preflight:
  [ok]   no earlier submission recorded locally for this task
  [FAIL] task_dryrun_unclaimed is not claimed. Run: submit_officer.py claim ...
REFUSED: 1 preflight error(s); nothing was sent.
exit=2
```

No traceback; no state files written.

### 7.3 File dry-run with metadata + sample CSV → PASS

Temporary fixture `work/tasks/task_dryrun_test/task.json`:

```json
{"id":"task_dryrun_test","inputType":"file","acceptedFileExtensions":"csv",
 "maxUploadedFileSize":10,"maxScore":100}
```

plus `sample_submission.csv` from the existing 800-row baseline. Command:

```bash
... submit-file --task-id task_dryrun_test \
  --file work/baseline_test/binary_tabular/submissions/submission.csv \
  --owner kozak --dry-run
```

```text
[ok] claim: task_dryrun_test -> kozak
[ok] inputType=file: file submission is valid
[ok] extension '.csv' accepted (task allows: csv)
[ok] file size 0.022 MB <= limit 10.00 MB
[ok] validator shape 800 rows x 2 cols
[ok] validator compared against work/tasks/task_dryrun_test/sample_submission.csv
[ok] validator PASS
request ... POST /submission/upload-file-url {fileSize:22645, fileName:"submission.csv", ...}
DRY RUN: no network request sent, no submission made.
exit=0
```

Afterwards: no `work/state/submissions/` directory and no
`work/state/submissions.jsonl`; the only state change was the claim.

### 7.4 Bad CSV → validator blocks (exit 4)

A 2-row CSV with a duplicate ID, an empty target and wrong row count produced:

```text
[FAIL] validator: row count 2 != sample 800
[FAIL] validator: 1 duplicate values in ID column 'row_id'
[FAIL] validator: column 'target': 1 NaN/empty value(s)
[FAIL] validator: column 'target' is constant (nunique=1)
REFUSED: 4 preflight error(s); nothing was sent.
exit=4
```

With `--allow-validation-fail` the same errors are downgraded to warnings and the
dry-run proceeds (exit 0); this flag is for emergencies only.

### 7.5 Input-type and prompt-limit guards

```text
# file task, submit-text
[FAIL] task inputType is 'file'; use submit-file, not submit-text
exit=2

# prompt task, 15,001 chars
[FAIL] prompt is 15001 chars > platform limit 15000
exit=2
```

### 7.6 `--yes` gate

A real-mode run on a claimed fake task printed the banner, preflight and request
preview, then:

```text
REFUSED: real submissions require --yes after checking the request above.
Nothing was sent.
exit=3
```

No `sigmoid_api.py` child was spawned; no backup/audit file was created.

### 7.7 Passed-task resubmit guard

Isolated state dir with a synthetic audit line `status:"passed"`:

```text
# dry-run: warning, still usable for inspection
[WARN] last local submission for task_guard PASSED ... A new one will replace it ...
DRY RUN: no network request sent ...

# real mode without --allow-resubmit
[FAIL] last local submission for task_guard PASSED ... Pass --allow-resubmit only with reviewer approval.
exit=2
```

### 7.8 Backup + audit smoke test (network boundary faked in-process)

To verify the post-`--yes` path without any HTTP, a temporary harness imported
`submit_officer`, replaced only `call_api()` with a stub returning
`{"status":"passed","finalScore":100}`, and ran a real-mode file submit against
an isolated state dir:

```text
SUBMIT_RC 0
BACKUPS ['20261114T092705Z_task_smoke_submission.csv']
BACKUP_BYTES_EQUAL True
AUDIT {'taskId': 'task_smoke', 'kind': 'file', 'owner': 'kozak',
       'status': 'passed', 'finalScore': 100, 'ok': True}
SMOKE_OK
```

This confirms backup naming/content and JSONL audit. It did **not** contact the
platform; the stub was the only network boundary. A text-mode smoke test also
verified that the exact text is saved as `<utc>_<task>_text.txt` and recorded
with `kind:"text"`.

### 7.9 Cleanup

Temporary task fixtures (`work/tasks/task_dryrun_test`,
`work/tasks/task_dryrun_prompt`), bad/long test files and `/tmp` state dirs were
removed. `status` reports `no claims yet`; `work/state/submissions.jsonl` does
not exist. Nothing was submitted to the live API.

---

## 8. Limitations / notes

- **No live API test was possible** (`allowSubmissions=false`, `/task` 403).
  The actual `sigmoid_api.py` call path is exercised only at the subprocess
  boundary; upload/finalize behavior is unchanged from the already-recon'd
  client.
- When `work/tasks/<task_id>/task.json` is absent, inputType/extension/size
  checks are skipped with a warning and the validator runs without a sample.
  Always run `tools/intake.py` after unlock before relying on preflight.
- `maxUploadedFileSize` units are inferred (MB for ≤2048, bytes otherwise)
  because the admin UI uses MB while some dumps may store bytes.
- Pending/passed guards use the local audit log, not a live API query. After a
  timeout/ambiguous API error, run `tools/sigmoid_api.py submissions`.
- `--no-claim`, `--force`, `--allow-resubmit`, `--allow-pending`,
  `--allow-validation-fail` and `--skip-validate` all exist for emergencies and
  print warnings; the normal path never needs them.

## 9. Next actions after unlock

1. Run `tools/intake.py` so every task has `task.json` + sample CSV.
2. Assign one owner per task with `claim`; record the owner in the war-room
   sheet.
3. For the first cheap task, use `--dry-run`, have a second teammate check the
   preview, then submit once with `--yes`.
4. Confirm evaluation status via `tools/sigmoid_api.py submissions`; watch for
   `pending` (team-wide lock) and never resubmit a passed task without
   `--allow-resubmit` + reviewer approval.
