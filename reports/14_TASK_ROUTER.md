# 14 — Task Router & Dashboard (offline triage layer)

_Subagent M, 2026-11-14 ~09:25–09:35 UTC. Offline only: no token, no live API, no submissions._

**Verdict: PASS** — `tools/task_router.py` routes all 3 mock tasks to the correct
modality/track (tabular, ctf/sec, cv), writes `plan.json` + briefs, and
`tools/task_dashboard.py` aggregates index/claims/submissions/plans into
`BOARD.md`. The repeatable verifier passes **40/40 checks**.

---

## 0. TL;DR for the orchestrator

When `/task` unlocks:

```bash
cd /home/kozak/mllm/MML
/home/kozak/mllm/py tools/intake.py                    # attachments -> work/tasks/<id>/files/
/home/kozak/mllm/py tools/task_router.py --all --force # plan.json + reports/levels/<id>.md
/home/kozak/mllm/py tools/task_dashboard.py --watch    # reports/levels/BOARD.md every 30 s
```

- Router is **offline** (stdlib-first; optional PIL/pyarrow only for media and
  Parquet introspection; no `sigmoid_api`, no network). It trusts intake's
  download records for success/404 status.
- Every task gets a `plan.json` with metadata, modality confidence and signals,
  fair-play extract, status (`ready/degraded/blocked`), track/template,
  baseline tool, four copy-paste commands, and a first 15-minute checklist.
- The dashboard shows the **latest** submission per task (platform scoring is
  latest, not best), owner, deadline and regression/blocked/pending alerts.
- Nothing is executed or submitted by either tool. They are decision-support.

---

## 1. What was built

### 1.1 `tools/task_router.py`

Pipeline per `work/tasks/<task_id>/task.json`:

1. **Discover** `*/task.json` under `--root` (default `work/tasks`), match
   `--task-id` by task id, directory name or country code.
2. **Metadata normalization** — `inputType` (text/file/prompt), country +
   continent, title, complexity, maxScore, timeLimitMinutes, realm/order,
   `gatewayForRealm`/isGateway, `ghUrl`, accepted extensions, max upload size.
3. **Links** — markdown `[label](url)` and bare URLs in `descriptionEn/Ro` +
   `ghUrl`; classifies attachment / repo / web / badge, extracts extension.
4. **Download reconciliation** — merges `task.json.downloadedFiles`,
   `_intake.files` and `urls.json[].download`; records 404s, failed and
   listed-not-downloaded entries. Files on disk are the final authority.
5. **File inspection** under `files/`:
   - extensions, sizes, SHA256 prefix (bounded budget), kind counts;
   - CSV/TSV: delimiter, header, row count (250k cap), per-column profile
     (numeric/text/path/categorical, uniqueness, image/audio path ratios), IDs,
     target candidates, sample-submission detection;
   - JSONL: keys, rows; Parquet: schema/rows via pyarrow if available;
   - images: folder counts, class groups (e.g. `train/alder`, `train/birch`),
     sampled dimensions/modes;
   - audio: counts, sampled WAV sample-rate/channels/duration;
   - archives, text corpora, model/checkpoint files.
6. **Modality inference** — weighted evidence:
   - file evidence first (3.0–4.6): text column -> `nlp`, image column/files ->
     `cv`, audio files -> `audio`, plain tables -> `tabular`;
   - description phrase patterns (0.35–3.6) for cv/nlp/audio/sec/ctf/rl/
     frontier/genai/tabular;
   - input type (`prompt` -> genai, `text` -> ctf) and continent prior (0.40);
   - outputs `modality`, `confidence`, ranked `scores`, top signals and
     alternatives; low evidence -> `unknown`.
7. **Status** — `ready` / `degraded` (supplementary 404) / `blocked` (no usable
   artifact for a file task, or all downloads failed).
8. **Fair play** — sentence-level extraction for pretrained models, external
   data and external APIs. Statuses: `allowed` / `forbidden` / `ambiguous` /
   `unspecified`, with quotes. Default is **deny**; generated commands never add
   `--pretrained` or `--allow-external-api`.
9. **Recommendation** — track, template, playbook, baseline tool, mission,
   `commands.{baseline,track,validate,submit}` and a 7-slot 15-minute checklist.
10. **Outputs** — `work/tasks/<id>/plan.json` and
    `reports/levels/<id>.md` (mock tests redirect this with `--brief-dir`).

`plan.json` shape (top-level keys):

```json
{
  "planVersion": 1,
  "sourceFingerprint": "…",
  "taskId": "task_…",
  "task": {"inputType": "…", "countryCode": "…", "maxScore": 100, "realm": "default", "…": "…"},
  "deadlineAt": "…Z", "hardStopAt": "…Z",
  "status": {"status": "ready|degraded|blocked", "reason": "…", "failed": [], "missing404": []},
  "modality": {"modality": "tabular", "confidence": 0.95, "scores": {}, "signals": [], "alternatives": []},
  "rules": {"metrics": [], "direction": "max|min|UNKNOWN", "flagHints": [], "submissionShape": "…"},
  "links": [], "downloads": {},
  "files": {"inventory": [], "tables": [], "images": {}, "audio": {}, "archives": [], "textCorpora": []},
  "recommendation": {
    "track": "…", "template": "…", "baselineTool": "…", "playbook": "…",
    "commands": {"baseline": "…", "track": "…", "validate": "…", "submit": "…"},
    "checklist": [{"time": "T+0–2", "step": "…", "detail": "…"}],
    "fairPlay": {}
  },
  "datasets": {"train": "…", "test": "…", "sample": "…", "target": "…", "idCol": "…"}
}
```

Brief (`reports/levels/<id>.md`) mirrors `reports/templates/TASK_BRIEF.md`:
snapshot, mission, rules extract (with fair-play quotes), data manifest with
hashes/rows/cols, recon findings, plan + full commands + checklist, validation,
risks, clock and artifacts. It is marked **router draft / not approved**.

Idempotency: a fingerprint of `task.json` + `urls.json` + file stat metadata is
stored in `plan.json`; an unchanged task is skipped unless `--force`.

### 1.2 `tools/task_dashboard.py`

Sources: `<root>/index.json`, `<root>/*/task.json`, `<root>/*/plan.json`,
`<state>/claims.json`, `<state>/submissions.jsonl`, optional
`<state>/journey.json`. Missing files are empty, so the tool runs before the
competition starts.

- Board columns: task/title, country, input type, modality, recommended track,
  owner, status, latest score/max, deadline.
- Status priority: latest submission (`PENDING/PASSED/REJECTED/ERROR`) >
  claim `BLOCKED`/`DONE`/`IN PROGRESS` > plan `BLOCKED`/`DEGRADED`/`READY` >
  `NEW`.
- Claims parser is tolerant: `{task_id: {owner,status,...}}`,
  `{claims:{...}}`, `{tasks:[...]}` or a list; owner aliases
  `owner/claimedBy/assignee/solver/user/member`.
- Submission parser accepts `taskId`/`task_id`/nested `task.id`, status and
  `finalScore|score|evaluatedScore`, timestamps; malformed lines are counted,
  not fatal. Latest is selected by timestamp (line order as fallback).
- Alerts: latest-score regression vs an earlier best, blocked/degraded tasks,
  pending evaluations, unclaimed tasks; 404 count included.
- `--json` emits the full snapshot; `--watch --interval 30` rewrites the board
  in a simple loop; `--iterations N` is a testing aid.

### 1.3 Safety / untrusted input

- No API/token imports and no network calls in either tool. URLs are classified
  from strings only; liveness comes from intake records.
- No raw task text is copied wholesale into reports. Descriptions are NFKC
  normalized, control/bidi/zero-width/Unicode-tag characters stripped, and
  truncated. CSV cell values and text corpus contents are **not** embedded —
  only headers, counts and stats. Artifacts remain passive data.
- Commands are default-deny for pretrained weights, external datasets and
  external APIs. CTF commands use static triage (`ctf_tactics.py`,
  `lmml_rs ctf-scan`) and never deserialize/execute unknown files.

---

## 2. Commands

### Production

```bash
cd /home/kozak/mllm/MML
PY=/home/kozak/mllm/py

$PY tools/intake.py                              # after /task unlocks
$PY tools/task_router.py --all --force           # all tasks, rebuild plans/briefs
$PY tools/task_router.py --task-id ar --force    # one task by country code
$PY tools/task_router.py --task-id task_mock_cv --json
$PY tools/task_dashboard.py                      # one board snapshot
$PY tools/task_dashboard.py --json               # machine-readable snapshot
$PY tools/task_dashboard.py --watch --interval 30
```

Both tools also accept `--root`, and the router accepts `--brief-dir` and
`--state-dir` (used by the mock suite to avoid touching live `reports/levels`).

### Mock suite (reproducible)

```bash
cd /home/kozak/mllm/MML
PY=/home/kozak/mllm/py

$PY work/router_test/make_fixtures.py
$PY tools/task_router.py --all --force \
  --root work/router_test --brief-dir work/router_test/briefs \
  --state-dir work/router_test/state --json > work/router_test/router_final.json
$PY tools/task_dashboard.py \
  --root work/router_test --state-dir work/router_test/state \
  --board work/router_test/BOARD.md --json > work/router_test/dashboard_final.json
$PY work/router_test/verify.py
```

Verifier output: `work/router_test/verify_output.txt` — **RESULT PASS, 40/40**.

---

## 3. Mock tasks

Fixtures live under `work/router_test/`; all URLs use `example.test` (nothing is
fetched). Briefs are written to `work/router_test/briefs/`, board to
`work/router_test/BOARD.md`.

| Task | Country | Input | Local files | Scenario | Expected modality | Router status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `task_mock_tabular` | MD | file | `train.csv` 120×6, `test.csv` 40×5, `sample_submission.csv` 40×2 | supplementary `extra_features.csv` **404** | tabular | degraded (core data OK) |
| `task_mock_flag` | BR | text | **none** (repo link only) | no downloaded files; flag + metadata description | ctf (track sec) | ready (text-solvable) |
| `task_mock_cv` | JP | file | 9 PNG (3 classes × 2 train + 3 test), 3 CSV | image folder + label CSV; `images.zip` **404** | cv | degraded (core data OK) |

The two required edge cases are covered: **no downloaded files** (flag task) and
**HTTP 404 links** (tabular + CV tasks). The CV task also exercises image-folder
class detection and label/image column detection.

Mock state (`work/router_test/state/`):

- claims: tabular `in_progress`, flag `blocked`, CV `in_progress`;
- submissions: CV `passed 84/150` then **latest `rejected 12/150`**; tabular
  `pending`; flag none;
- journey deadlines: 14:00 / 14:30 / 15:00 UTC.

---

## 4. Test results

### 4.1 Router

`work/router_test/router_final.json`: 3 selected, 3 processed, 0 errors.

| Task | Modality (confidence) | Track | Template | Status | Deadline / hard stop |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `task_mock_tabular` | **tabular** (0.95, strong) | tabular | `templates/02_tabular_fast.py` | degraded | `11-14 14:00Z` / `13:48Z` |
| `task_mock_flag` | **ctf** (0.91, strong) | sec | `templates/05_adversarial_sec.py` (forensics) | ready | `11-14 14:30Z` / `14:18Z` |
| `task_mock_cv` | **cv** (0.89, strong) | cv | `templates/03_cv_fast.py` | degraded | `11-14 15:00Z` / `14:48Z` |

Evidence and generated outputs confirmed by the verifier:

- Tabular: detected 3 tables, `target=churn`, `idCol=customer_id`, metric
  `roc-auc/max`, 404 recorded, sample shape 40×2; baseline = `make_baseline.py`,
  track = `02_tabular_fast.py`, validate/submit commands populated.
- Flag: `files.count=0`, 1 repo link, no tables; flag hint `flag{...}`; fair play
  `pretrained=forbidden`, `externalData=forbidden`, `externalApi=forbidden`;
  static triage commands emitted; no fake CSV submission path.
- CV: `classGroups={"train":{"alder":2,"birch":2,"cedar":2}}`, image column
  `filename`, `target=label`, 9 images, 404 `images.zip`; baseline =
  `make_baseline.py`, track = `03_cv_fast.py` with `--data-dir`,
  `--train-csv`, `--test-csv`, `--sample-sub`, image/target/id columns.
- Each plan has 7 checklist slots and all four commands; each brief is
  8.6–10.6 KB and contains the data manifest, fair-play quotes and clock.

Brief spot-checks: `work/router_test/briefs/task_mock_tabular.md`,
`task_mock_flag.md`, `task_mock_cv.md`.

### 4.2 Dashboard

`work/router_test/dashboard_final.json` + `work/router_test/BOARD.md`:

| Task | Owner | Status | Score | Deadline | Alert |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `task_mock_flag` | solver-ctf | **BLOCKED** | — | 14:30Z | claim note "artifact unavailable; awaiting re-download" |
| `task_mock_tabular` | solver-tabular | **PENDING** | pending | 14:00Z | team-wide evaluation lock |
| `task_mock_cv` | solver-cv | **REJECTED** | **12/150** | 15:00Z | latest 12 replaced earlier 84 — latest counts |

Counter: `{'BLOCKED': 1, 'PENDING': 1, 'REJECTED': 1}`. All owners populated;
all counts, deadlines and alerts correct. This specifically validates that the
board reports the latest attempt, not the best.

### 4.3 Edge tests

| Test | Result |
| :--- | :--- |
| Router second run without `--force` | 3 skipped, 0 processed, 0 errors (fingerprint cache) |
| `--task-id br --force` | 1 selected / 1 processed (`task_mock_flag`) |
| `--watch --interval 1 --iterations 2` | 2 snapshots, exit 0 (`work/router_test/watch.log`) |
| Real `work/tasks` (API locked, empty) | router: 0 tasks, exit 0; dashboard: 0 tasks, empty `reports/levels/BOARD.md`, exit 0 |
| Scratch prompt task (`/tmp/opencode`) | modality genai 0.95, track genai, prompt draft/submit commands, no crash |
| Scratch text-answer task with CSV (`/tmp/opencode`) | modality tabular 0.86, track tabular, intermediate CSV commands + `submit-text`, no file-upload command |
| Malformed/absent state files | dashboard starts empty; malformed JSONL lines are counted |
| URL query redaction (scratch task with `?X-Amz-Signature=…`) | brief contains the path only; secret query string absent; raw URL stays in gitignored `plan.json` |

No solver template was executed on the mock data (routing/dashboard test only),
so no CPU was spent and no claim about template accuracy is made.

---

## 5. Limitations / open items

1. **Heuristics, not ground truth.** Modality is weighted evidence. Ambiguous
   tasks (e.g. "classify images with metadata") still get alternatives and a
   confidence; weak evidence yields `unknown`. Re-run with `--force` after
   reading the task or downloading more files.
2. **No link liveness checks.** A 404 is reported only if intake recorded it.
   If intake is stale, run `tools/intake.py --task-id <id> --force` first.
3. **No hints / Frozen journey.** Hint penalties and `deadlineAt` are unknown
   unless `work/state/journey.json` exists (the router reads it when present).
   `/hint/task-overview` is never called.
4. **Inspection depth.** CSV/TSV/JSONL and Parquet headers are parsed; XLSX
   cell contents, JSON array internals, archive members and PDF text are not.
   Image dimensions are sampled (15), WAV durations sampled (20). Non-WAV audio
   duration is not decoded.
5. **No raw content in reports.** CSV cells/text corpora are deliberately not
   embedded (prompt-injection defense), so a solver must open the data for
   content-level clues. Briefs also redact URL query strings/fragments; raw
   links remain only in gitignored `work/tasks/<id>/plan.json` for intake
   reconciliation.
6. **Fair-play extraction is sentence-level regex.** Conditional text such as
   "no pretrained models unless explicitly allowed" is marked `ambiguous`, not
   `forbidden`; the default remains deny and the orchestrator must read the
   quote. Never rely on the router to authorize `--pretrained` or external APIs.
7. **Commands are recommendations.** Placeholders appear when detection is
   incomplete: `TRAIN_CSV_PATH`, `TEST_CSV_PATH`, `SAMPLE_CSV_PATH`, `ENV_ID`,
   `NODES_CSV_PATH`, `EDGES_CSV_PATH`, `CORPUS_FILE`, `FLAG_VALUE`,
   `ANSWER_VALUE`. Replace them before running. The router never executes them.
8. **Brief filename conflict.** Per this assignment the router writes
   `reports/levels/<task_id>.md`; the war-room template reserves the same path
   for the final `TASK_REPORT`. The brief is explicitly marked "router draft";
   the orchestrator should archive it (e.g. `<id>_ROUTER_BRIEF.md`) before the
   scribe writes the closeout report.
9. **Dashboard schemas are tolerant but not canonical.** There is no live
   claims/submissions writer yet; the parser accepts common key variants. The
   team should standardize on `taskId`, `status`, `finalScore`, `submittedAt`
   in `submissions.jsonl`, and `{task_id: {owner,status,deadlineAt,note}}` in
   `claims.json`.
10. **`--watch` is a simple timer loop** (default 30 s), not inotify; it rewrites
    the board every tick and does not detect file events.
11. **Country/continent map** covers the 54 codes observed in recon; unknown
    codes get `Unknown` and no continent prior.
12. **No GPU and 4 CPU cores** are assumed in commands (`OMP_NUM_THREADS=2`,
    no pretrained flags, CPU baselines). Modal is not invoked by the router.

---

## 6. Files

| Path | Role |
| :--- | :--- |
| `tools/task_router.py` | offline task triage, `plan.json` + brief writer |
| `tools/task_dashboard.py` | index/claims/submissions/plans -> `reports/levels/BOARD.md` |
| `work/router_test/make_fixtures.py` | 3 mock tasks + mock state |
| `work/router_test/verify.py` | 40-assertion mock verifier |
| `work/router_test/task_mock_*/plan.json` | generated plans |
| `work/router_test/briefs/task_mock_*.md` | generated briefs (mock-only dir) |
| `work/router_test/BOARD.md` | generated mock board |
| `work/router_test/router_final.json`, `dashboard_final.json`, `verify_output.txt`, `watch.log` | test evidence |
| `reports/levels/BOARD.md` | live board (currently 0 tasks, API locked) |
| `reports/14_TASK_ROUTER.md` | this report |

**Not touched:** tokens, `.env`, live API, `templates/`, `optimizations/`,
`lvl_XX/`; no git add/commit/push.

---

## 7. Next actions at unlock

1. Run intake, then `task_router.py --all --force`; inspect all `unknown` /
   `ambiguous` / `blocked` tasks first.
2. Spawn solvers from the briefs; the 15-minute checklist is the shared clock.
3. Have the orchestrator write claims with one owner per task and update
   `submissions.jsonl` after every attempt.
4. Run `task_dashboard.py --watch` in the war room; act on regression, blocked
   and pending alerts immediately (latest counts; pending blocks the team).
5. Re-run `intake.py --task-id <id> --force` + `task_router.py --task-id <id>
   --force` whenever a missing/failed download is retried or a task description
   changes.
