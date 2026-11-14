# 01 — Site & API Recon (LMML 2026 / sigmoid.academy)

_Agent A (API & task-intake), 2026-11-14 ~08:54 UTC. All live probes were read-only._

Legend: **[V]** = verified against the live API or live SPA bundle; **[I]** = inferred from the SPA bundle / API conventions, not yet exercised (task API still locked).

## 1. Base URL and auth model

- **API base [V]:** `https://api-prod.sigmoid.academy/api/v1/`
- **SPA [V]:** `https://sigmoid.academy/` (bundle `assets/index-Csrl3MFr.js`, sha256 `ce2dbcf…82fe1`; copy at `/tmp/opencode/sigmoid-app.js`).
- **Auth [V]:** JWT (HS256) in browser `localStorage["auth-token"]`, sent as `Authorization: Bearer <jwt>`. The SPA attaches it to every endpoint except the public auth/settings calls and logs the user out on HTTP 401.
- **Token location:** `/home/kozak/mllm/.sigmoid_token` (mode 600, outside the repo). Never print, commit, or pass it in URLs.
- **Token claims (decoded locally, token not printed) [V]:** `sub=111`, `role=user`, `aud=sigmoid.academy`, issued `2026-11-14T08:05:44Z`, expires `2026-11-15T08:05:44Z` (24 h). Refresh the file before then.
- **Other headers:** `Accept: application/json`; JSON bodies `Content-Type: application/json`; presigned uploads use a plain `PUT` with the file bytes and no bearer header.
- **Unauthenticated endpoints used [V]:** `GET /settings`; `POST /auth/login|register|...`.
- **Live updates [I]:** Socket.IO at `https://api-prod.sigmoid.academy/socket.io/`; events include `appPhase:update`, `settings:updated`, `leaderboard:update`, `submission:evaluated`, `clue:unlocked`, `team:realmJourneyUpdated`. Our poller does not depend on it.

## 2. Current competition state (verified 2026-11-14 08:47–08:52 UTC)

| Probe | Result |
|---|---|
| `GET /settings` | `{"allowSignUp":true,"allowSubmissions":false}` |
| `GET /realm-journey/my` | `{"serverNow":"2026-11-14T08:51:39.176Z","journey":null}` |
| `GET /task?limit=100&offset=0` | **HTTP 403** `{"message":"task access is denied","error":"Forbidden","statusCode":403}` |
| `GET /task/countries` | array of **54** country codes (`eg, ee, et, fj, al, ao, ar, au, …`) — not the task list |
| `GET /clue/passphrase` | 10 slots, `unlockedCount=0`, `complete=false`; slot task ids: `task_argentina`, `task_australia`, `task_belgium`, `task_chile`, `task_france`, `task_japan`, `task_new_zealand`, `task_norway`, `task_south_african_republic`, `task_united_kingdom` |
| `GET /submission/team/my?limit=1` | `{"totalItems":0,"items":[]}` |
| `GET /scoring/leaderboard` | 44 teams, all scores 0 |
| `GET /hint/task-overview/task_argentina` | **HTTP 403** (same task lock) |
| `/auth` (earlier recon) | user `Kozak` id 111, team **HardCoders** id 37, realm `default` |

**Conclusion:** journey has not started; tasks are hidden. Our tools degrade to a “locked” state and exit 0.

## 3. Endpoints relevant to tasks / submissions / hints / clues / scoring

### Tasks
| Endpoint | Status | Notes |
|---|---|---|
| `GET /task?limit=&offset=` | [V] exists, locked now | Task list. Response `{items:[Task], totalItems:n}` **[I]**. SPA calls `{offset:0,limit:100}`. |
| `GET /task/:identifier` | [V] exists, locked now | Single task. SPA passes **country code** (`/task/ar`), not `task_argentina` **[V]**. Also accepts the task id by convention **[I]**. |
| `GET /task/countries` | [V] | Country-code array used for flag/continent UI. |
| `GET /task/json`, `GET /task/csv` | [I] | Admin export; `POST` uploads. |
| `GET /settings` | [V] | `{allowSignUp, allowSubmissions, submissionsWillOpenAt?, submissionsWillCloseAt?}`; dates ISO strings **[I]**. `POST /settings` is admin. |

**Task schema [I, from SPA admin form + list rendering]**

```json
{
  "id": "task_argentina",
  "countryCode": "ar",
  "titleEn": "…", "titleRo": "…",
  "descriptionEn": "markdown …", "descriptionRo": "markdown …",
  "maxScore": 100,
  "complexity": "easy|medium|hard",
  "inputType": "text|file|prompt",
  "maxUploadedFileSize": 10485760,
  "acceptedFileExtensions": "csv,py;ipynb or *",
  "ghUrl": "https://github.com/… or null",
  "realm": "default|frozen",
  "realmOrder": 1,
  "timeLimitMinutes": 120,
  "gatewayForRealm": "default|frozen|null",
  "createdAt": "ISO-8601"
}
```

Attachments/data are not a separate array in the form: they appear as URLs inside `descriptionEn/Ro` (markdown links) or `ghUrl`; `tools/intake.py` scans the whole JSON recursively.

### Journey (24 h chain)
| Endpoint | Status | Notes |
|---|---|---|
| `GET /realm-journey/my` | [V] | `{serverNow, journey:null}` now. When running, `journey` has `realm`, `status`, `tasks[]` **[I]**. |
| `POST /realm-journey/my/leave` | [I] | Leave current realm. |
| `DELETE /realm-journey/team/:teamId/:realm` | [I] | Admin reset. |

**Journey object [I, from SPA]** — `tasks[]` entries: `{taskId, countryCode, deadlineAt, status:"active|completed|left", passedAt}`. SPA counts `passedAt` and computes remaining from `deadlineAt` vs `serverNow`.

### Submissions
| Endpoint | Method | Notes |
|---|---|---|
| `POST /submission/text` | POST | Body `{taskId, text, url?}`. Used for **text (flag)** and **prompt (LLM chat)** tasks **[V]** from SPA code. |
| `POST /submission/upload-file-url` | POST | Body `{taskId, fileSize, fileName, fileExtension, contentType?}` → `{allowed, url, key, errorKey?}` **[V/I]**. `fileName` excludes the extension; `fileExtension` has no dot. SPA omits `contentType`; `sigmoid_api.py` adds it. |
| presigned `PUT url` | PUT | Raw file body, `Content-Type: <mime>`, **no** bearer header **[V]**. |
| `POST /submission` | POST | Body `{taskId, fileStorageKey, url?}` after a successful PUT **[V]**. |
| `GET /submission/my` | GET | Current user's submissions. |
| `GET /submission/team/my?limit=&offset=` | GET | Team submissions; `{totalItems, items}` **[V]**. |
| `GET /submission/team/:teamId` | GET | Same for a team. |
| `GET /submission/team/my/pending` | GET | Task ids with a pending evaluation; returns an array **[V]** (`[]` now). |
| `PATCH /submission/:id/accept|reject`, `DELETE /submission/:id` | admin | Moderation. |
| `GET /submission/admin/:id/file-url` | GET | Admin download URL. |

**Submission schema [I, from admin table + submit handler]**

```json
{
  "id": 1, "taskId": "task_argentina", "teamId": 37, "userId": 111,
  "status": "pending|passed|rejected|error",
  "text": "…", "url": "…", "fileStorageKey": "…",
  "score": 0, "finalScore": 0, "penalty": 0, "evaluatedScore": 0,
  "details": "evaluation message",
  "createdAt": "ISO-8601",
  "task": {"id":"…","countryCode":"ar","titleEn":"…","maxScore":100},
  "team": {"id":37,"name":"HardCoders"}, "user": {"id":111,"username":"Kozak"}
}
```

After `POST`, the SPA switches on `status`: `pending` → “queued”, `passed` → success with `score`, `rejected` → not passed with `score`, anything else → evaluation failed. `/clue/passphrase` unlocks a word when a task passes (Antarctica meta-puzzle).

### Hints
| Endpoint | Method | Notes |
|---|---|---|
| `GET /hint/task-overview/:taskId` | GET | Hints for a task: text EN/RO, `penalty`, locked/unlocked state **[I]**. |
| `GET /hint/locked/:taskId` | GET | Locked hints overview. |
| `GET /hint/unlocked/:taskId` | GET | Team's unlocked hints. |
| `POST /hint/unlock/:taskId` | POST | Unlock next hint; no body in SPA; costs `penalty` points **[V/I]**. |
| `GET /hint?limit=&offset=&taskId=&search=` | GET | Admin list. |
| `POST /hint`, `PATCH|DELETE /hint/:id`, `POST /hint/json` | admin | CRUD/import. |

### Clues / scoring / team
| Endpoint | Method | Notes |
|---|---|---|
| `GET /clue/passphrase` | GET | `{slots:[{slot,word,taskId,countryCode,unlockedAt,trigger}],complete,unlockedCount,total}` **[V]**. |
| `POST /clue/admin/teams/:teamId/grant|revoke`, `/reset`, `GET /clue/admin/overview` | admin | Manual clue management. |
| `GET /scoring/leaderboard` | GET | `{totalItems,items:[{id,name,score,isInternal,members:[…]}]}` **[V]**. |
| `GET /scoring/team/:teamId` | GET | Team score detail. |
| `GET /scoring/leaderboard/:teamId/details` | GET | Per-task breakdown **[I]**. |
| `GET /team/my`, `GET /team/my/country-progress` | GET | Team + completed-country progress; `[]` now **[V]**. |

## 4. Submission flows (as implemented by the SPA)

**Text / flag**
1. `POST /submission/text` `{taskId:"task_<country>", text:"flag{…}", url:null}`.
2. Response `status` decides pending/passed/rejected; `score` is the awarded score.

**Prompt (LLM chat)**
1. Same `POST /submission/text`; `text` is the player's prompt turn.
2. Chat history is the team's submissions for that task (`GET /submission/team/my?taskId=<id>&limit=100`) **[I]**.
3. Evaluator replies/updates the submission asynchronously; `GET /submission/team/my/pending` lists tasks with pending evaluation.

**File**
1. `POST /submission/upload-file-url` `{taskId, fileSize, fileName:"answer", fileExtension:"csv"}`.
2. If `allowed=false` or missing `url`/`key` → show `errorKey`/generic error (size or extension rejected).
3. `PUT <url>` with raw bytes and `Content-Type` from the browser file type.
4. `POST /submission` `{taskId, fileStorageKey:<key>, url:null}`.
5. Same status/score handling as text.

Local helpers already implement all three: `tools/sigmoid_api.py`
(`submit-text`, `submit-file`, `request_upload`, `upload_file`). Do **not** submit while
`allowSubmissions=false` / journey is null.

## 5. Delivered tooling

### `tools/intake.py`
- Fetches `GET /task` with pagination (`limit`, default 100), saves raw JSON to
  `work/tasks/_raw/tasks_<UTC>.json` + `latest.json`.
- Writes `work/tasks/<task_id>/task.json` (normalized copy + `_intake` metadata),
  `summary.md` (human title/description/links/files), `urls.json`, and downloads
  attachments to `work/tasks/<task_id>/files/`.
- Recursive URL discovery handles absolute URLs, markdown links, `/relative`,
  `./relative`, protocol-relative, and presigned URLs. Bearer auth is attached only
  to API-origin URLs and as an **unredirected** header; presigned URLs get no auth.
  Non-200/timeouts/oversize/empty bodies are recorded in the manifest and skipped.
- Writes `work/tasks/index.json` and `reports/levels/INDEX.md` (id, country, title,
  complexity, inputType, time limit, max score, files).
- `--dry-run` (no writes/downloads), `--task-id` (repeatable/comma-separated,
  accepts id or country code), `--force`, `--no-download`, `--max-file-mb`, `--json`.
- Locked API: prints `LOCKED: HTTP 403 …`, records `_raw/locked_<UTC>.json`, writes a
  placeholder `reports/levels/INDEX.md` **only if no real index exists**, exits 0.

### `tools/watch_state.py`
- Polls `GET /settings`, `/realm-journey/my`, `/clue/passphrase`,
  `/submission/team/my?limit=1` every 20 s (`--interval`).
- When `allowSubmissions=true` or `journey!=null`: calls `GET /task` through the
  intake module immediately, then refreshes every 5 min (`--intake-interval`).
- Appends every poll/transition to `work/state/events.jsonl`; rewrites
  `reports/LIVE_STATUS.md` (UTC time, settings, journey summary, task count, clue
  progress, submission count).
- HTTP 401 → loud `!!! ALERT … TOKEN EXPIRED OR INVALID … refresh
  /home/kozak/mllm/.sigmoid_token !!!` in `work/state/watch.log` and at the top of
  `reports/LIVE_STATUS.md`.
- `--once`, `--max-cycles N`, `--stop-file PATH` (default `work/state/stop`),
  `--clear-stop`; graceful SIGINT.

## 6. Tests run (locked API, 2026-11-14 08:50–08:54 UTC)

- `intake.py --dry-run` → `LOCKED: HTTP 403`, exit 0, **no files written**.
- `intake.py` → locked raw JSON + placeholder `reports/levels/INDEX.md`, exit 0.
- `watch_state.py --once` → one poll event, `LIVE_STATUS.md` written, exit 0.
- `watch_state.py --interval 1 --max-cycles 2` and `--stop-file` → clean stop, exit 0.
- Injected `SIGMOID_TOKEN=invalid.test.token` → one consolidated HTTP 401 ALERT
  line (naming all three failing endpoints) in `work/state/watch.log` + loud
  banner in `reports/LIVE_STATUS.md`; rerun with the real token restored
  “Auth: OK”.
- Unlock-path simulation (monkeypatched settings/journey in memory) → detected the
  transition, called intake, handled the still-locked `GET /task` 403 without crash.
- Synthetic task normalization in `/tmp` → valid `task.json`, `summary.md`,
  `index.json`, `INDEX.md`; local HTTP test of the downloader → 200 saved, 404
  recorded as failed without exception.
- Token leak scan over `work/`, `reports/`, and both tools using the real token
  value → **no matches**.

## 7. Exact commands

```bash
cd /home/kozak/mllm/MML
PY=/home/kozak/mllm/py

# state probes
$PY tools/sigmoid_api.py settings
$PY tools/sigmoid_api.py journey
$PY tools/sigmoid_api.py tasks
$PY tools/sigmoid_api.py clue
$PY tools/sigmoid_api.py submissions
$PY tools/sigmoid_api.py leaderboard

# intake
$PY tools/intake.py --dry-run
$PY tools/intake.py
$PY tools/intake.py --task-id task_argentina --force
$PY tools/intake.py --task-id ar --no-download

# watcher
$PY tools/watch_state.py --once
$PY tools/watch_state.py --interval 20
touch work/state/stop           # stop a running watcher

# SPA bundle (for offline recon)
curl -sL https://sigmoid.academy/ -o /tmp/opencode/sigmoid-index.html
curl -sL https://sigmoid.academy/assets/index-Csrl3MFr.js -o /tmp/opencode/sigmoid-app.js
```

## 8. Blockers / next actions

1. **Blocker:** tasks and hints stay 403 until the journey starts (`allowSubmissions`
   true or `journey` non-null). Watcher is ready to trigger automatically.
2. Token expires `2026-11-15T08:05:44Z`; on HTTP 401 refresh
   `/home/kozak/mllm/.sigmoid_token` and restart the watcher.
3. Once unlocked, confirm whether `GET /task/<countryCode>` returns the same object
   as the list id (`task_<country>`); intake uses the list `id` for directories.
4. Verify the actual `upload-file-url` response and submission statuses on the first
   real file submission; schemas above are SPA-derived until then.
5. Attachment downloads were exercised only against a local HTTP server; real task
   URLs (including presigned storage) remain untested until tasks unlock.
