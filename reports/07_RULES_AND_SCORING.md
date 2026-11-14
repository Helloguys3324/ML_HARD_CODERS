# 07 — Competition Rules & Scoring Recon (LMML 2026)

_Agent H, 2026-11-14 08:30–09:00 UTC. Read-only probes only; no submissions were made._

**Evidence base**

- SPA bundle `https://sigmoid.academy/assets/index-Csrl3MFr.js` (sha256 `ce2dbcf…82fe1`, copy `/tmp/opencode/sigmoid-app.js`), map chunk `Map-CTBDek3T.js`, and the full English i18n catalogue `https://sigmoid.academy/locales/en/translation.json` (saved to `work/rules/en.json`).
- Live API `https://api-prod.sigmoid.academy/api/v1/` with the team JWT (never printed).
- Public pages captured with Playwright (`work/rules/page_*.txt|png`), plus the JSON-LD Event block in `https://sigmoid.academy/` (`/tmp/opencode/sigmoid-index.html`).
- Legend: **[V]** = directly observed in bundle/API/live page; **[I]** = inferred from UI/server behaviour, must be confirmed live.

---

## 0. TL;DR for the orchestrator

1. **It is a team competition.** The public leaderboard ranks **teams only**. `GET /scoring/team/37` returns `{"score":0}`; there is no individual-score endpoint.
2. **The latest submission per task counts for the team — not the best.** The admin CSV selector says verbatim: *"Checked: only the latest submission per task, the ones that add up to the team score."* **[V]** Deleting the counting submission "falls back" to the previous submission. So a later worse/failed attempt can replace a better one; coordination is critical.
3. **Rejected ≠ zero.** The UI has *"Submission not passed ({{score}} pts)"* and the admin submission table has separate `finalScore`, `penalty`, `evaluatedScore` columns. Partial credit exists. **[V]**
4. **Hints cost points and are irreversible.** *"You lose {{penalty}} points by using the hint."* There is no cooldown, but unlocking after the task deadline is forbidden. **[V]**
5. **The Antarctica meta-puzzle is real and separate.** 10 clue slots (10 named country tasks) produce a 10-word passphrase. Completing it makes the team *"eligible for Frozen"*. Frozen is a **5-stage timed chain** (PR #271 seeding) entered through a hidden **gateway** task. **[V/I]**
6. **Platform-level fair play is generic.** The only anti-cheat text on the platform is the Terms & Conditions (cheating, unauthorized access, stealing/sharing flags/solutions, disqualification). There is **no global "no pretrained models / no external APIs / no external datasets" string** in the SPA or i18n. Those restrictions, if any, live in each task's Markdown description. **[V]**
7. **The event has not started yet.** At 2026-11-14 08:54 UTC: `allowSubmissions=false`, `journey=null`, `GET /task` → 403. The only dates on the site are in JSON-LD, and they conflict with the "24h" branding — treat them as approximate and poll `/settings`. **[V]**
8. **Do not assume "24 levels".** `GET /task/countries` currently returns **54 country codes**, and tasks are addressed by **country code** (`GET /task/ar`), effectively one task per country. The real task count is `GET /task.totalItems` once it opens. "24h" is the duration. **[V/I]**

---

## 1. Verified facts

### 1.1 Scoring model

**Endpoints [V]**

| Endpoint | Response (observed / SPA-derived) |
|---|---|
| `GET /scoring/leaderboard` | `{totalItems, items:[{id,name,score,isInternal,members:[{id,username}]}]}`. 44 teams, all `score:0` now. |
| `GET /scoring/team/:teamId` | `{score: number}`. `GET /scoring/team/37` → `{"score":0}`. |
| `GET /scoring/leaderboard/:teamId/details` | `{teamId,teamName,isInternal,totalTeamSubmissions,passedTeamSubmissions,rejectedTeamSubmissions,members:[{userId,username,totalSubmissions,passedSubmissions,rejectedSubmissions,scoreContributed}]}` |
| `GET /submission/admin/team-scores/csv?teamId=&onlyCounting=` | CSV; admin only (403 for us). Non-counting export includes a `counts_toward_score` column. |

**What the UI says [V]**

- Leaderboard columns: `Placement | Name | Members | Score`. Placement is the array index + 1; the client **does not sort** — server order is authoritative. All-zero teams get distinct placements (#1..#44) because the client numbers rows; whether the server order encodes an intentional tie-break is unknown.
- Expanded team shows per-member `Submissions done / passed / not passed / Score contributed` (`scoring.submissionsDoneCount`, `submissionsPassedCount`, `submissionsRejectedCount`, `scoreContributed`). Team total row shows `-` for score contributed, i.e. the team score is the sum of member contributions **[I]**.
- Leaderboard polls every **10 s** (`Mu=1e4`) while the tab is focused; also refreshes on `leaderboard:update` socket events.
- Internal (organizer/Sigmoid) teams are tagged `(Sigmoid team)` and still appear in the ranking.

**Per-task score fields [V]**

- Submission list shows `finalScore / task.maxScore`.
- Admin submission table shows, in order: `finalScore`, `penalty`, `evaluatedScore`, plus `status`, `details`.
- `submission.evaluatedPassed = "Submission for {{task}} passed ({{score}} pts)"`, `evaluatedNotPassed = "Submission for {{task}} not passed ({{score}} pts)"`.
- `submission.notPassedWithScore = "Submission not passed ({{score}} pts)"` — **rejected submissions can still score**.
- Task admin hint for `maxScore`: *"Shown as the task label."* Default is **100**, integer ≥ 1. So `maxScore` is a display/label value; the evaluator returns `evaluatedScore`, and `finalScore` is the net number used for scoring **[I]**.

**Counting rule [V]**

> *"Checked: only the latest submission per task, the ones that add up to the team score. Unchecked: every submission the team ever made, with a counts_toward_score column telling them apart."* — admin "Download team score submissions" dialog.

> *"Submission #ID deleted. The team's score for the task fell back to submission #PREV (N points)."* / *"…it wasn't the one counting, the team's score is unchanged."* — admin delete toasts.

Consequences:
- Scoring is **team-scoped and task-scoped**: one counting submission per task for the whole team (any member).
- A new submission — passed or rejected — becomes the candidate counting one. The platform text says **latest**, not best. **A failed/worse latest attempt can lower the team score** **[I, high confidence — verify on first task]**.
- Need to know `status === passed` for **realm completion** too: *"A realm only counts as completed when the team's latest submission of every one of its tasks passed."* This independently confirms "latest" semantics at the realm level. **[V]**

**Partial credit / max score**

- File/evaluator tasks can return any `evaluatedScore`; rejected can be non-zero.
- Nominal maximum is `Σ task.maxScore` across tasks, but whether the server caps `finalScore` at `maxScore` is not visible in the SPA. **[open]**
- No evidence of score normalization, per-task weighting, or a time bonus.

**Team vs user score**

- Only team score is ranked. Member `scoreContributed` is informational and is almost certainly the sum of that member's counting submission `finalScore`s. Sum(member contributions) = team score **[I]**.
- Team capacity is **5** (`members.length >= 5` disables join/invite; toast `Team is full`). Our team has 2 members.
- `leaderboardDescription` says *"See how users and teams rank…"* but no user leaderboard route/endpoint exists; this string is stale. **[V]**

**Tie-breaking**

- Client trusts server order. Current all-zero order is `Mugiwara, a, Minimax, Dreams, …`; it matches neither team id, name, nor `createdAt` order (checked team ids 2/3/5/6). The tie-break rule is server-side and **not exposed**. **[open]**

**Corrections to `reports/01_SITE_API_RECON.md`**

- Hint unlock is `POST /hint/unlock/:hintId` — the path parameter is the **hint id**, not the task id.
- `GET /scoring/leaderboard/:teamId/details` is **not** a per-task breakdown; it returns team submission counts and per-member `scoreContributed`.
- Prompt chat history fetched by the SPA is the last **20** submissions (`Pc=20`), not 100.
- Task IDs are `task_<country_name>` but the participant API route uses the **two-letter country code** (`GET /task/ar`).

---

### 1.2 Time limits and journey progression

**Task fields [V]** (admin task form)

| Field | Default | UI hint / meaning |
|---|---|---|
| `timeLimitMinutes` | **120** | *"How long the task takes submissions once unlocked. Changing it does not move deadlines already running."* Integer > 0. |
| `realmOrder` | null | *"Tasks of a timed realm unlock one at a time, lowest first. Empty goes last, by creation time."* Integer ≥ 0 / null. |
| `realm` | `default` | `default` or `frozen`. |
| `gatewayForRealm` | null | *"Passing this task moves the team into this realm."* `default`, `frozen`, or none. |

**Journey endpoint [V]**

`GET /realm-journey/my` → `{serverNow: ISO-8601, journey: null | Journey}`.

Journey fields used by the SPA:
- `journey.status`: `active | completed | left`.
- `journey.tasks[]`: `taskId`, `countryCode`, `status` (`active | completed | left`), `deadlineAt` (ISO), `passedAt` (ISO/null), plus display fields (`titleEn`, `titleRo`). **[I on exact field set]**
- `serverNow` is authoritative: the client computes `offset = serverNow − responseArrivalTime`, then uses `Date.now() + offset` for all deadline checks (1 s tick). Our server clock matches the host within ~1 s.

**Where the timer is enforced [V/I]**

- The journey hook **skips the API entirely unless `team.realm === frozen`**: `t = !!user?.team && team.realm === 'frozen'`. Therefore:
  - **Default realm tasks have no visible per-task countdown** and no journey in the UI. They appear to be open for the whole submissions window; `timeLimitMinutes` on Default tasks is probably inert. **[I, high confidence]**
  - **Frozen realm tasks** get a `deadlineAt` from the server when unlocked. This is the only place `timeLimitMinutes`/`deadlineAt`/`taskTimeExpired` are used.
- Client timer format `H:MM:SS`; it turns red under **10 minutes** (`qc = 600000` ms). The floating journey HUD also warns at < 10 min.
- Expired task: submit button disabled, *"Time is up, this task no longer accepts submissions."*; server rejects with **403 `taskTimeExpired`**. Hint unlock is blocked too (`realm.hintTimeExpired`).
- Other task-access 403 codes: `taskNotUnlocked` (future/sequential task), `realmLeft` (team already left the realm), `taskNotInRealm` (task belongs to another realm).

**Journey end [V]**

Modal text:

> *"Your journey has ended"*
> *"Time is up on every task you unlocked. You passed {{passed}} of {{total}}. Your team can now only leave the realm."*
> Buttons: `Review tasks`, `Leave realm`.

Leaving:

> *"This cannot be undone. Your team leaves the realm now and will never be able to enter it again. Tasks you have not finished stay unfinished."* (requires typing the team name to confirm)

**Realm completion [V]**

> *"A team leaves a realm by passing all of its tasks, then it returns to the Default realm. A realm only counts as completed when the team's latest submission of every one of its tasks passed."*

**24h vs per-task limits**

- The event is branded **24h** (`"Be ready to start your 24h Journey"`, title `LMML 2026 - 24-Hour Machine Learning Marathon`). The 24h is the global submissions window.
- Per-task `timeLimitMinutes` (default 120) applies to the **Frozen chain**, not to the Default 24h window **[I]**.
- Submissions open/close globally via `allowSubmissions`; there is no automatic global deadline in the app. See §1.7.

**Can teams skip / replay?**

- **Default:** yes to replay (resubmit any time while open); no sequential gating; all tasks visible. Replaying a passed task is risky under latest-counts semantics.
- **Frozen:** tasks unlock **one at a time, lowest `realmOrder` first**; no skipping. Re-submission while a stage is active is allowed; pending evaluation blocks it. A passed stage unlocks the next one. If the active stage deadline expires before it is passed, the journey ends (no next unlock) — **verify on first Frozen stage**. Leaving is permanent.
- **Replay after leaving:** not possible; `realmLeft`. Only an admin reset (`DELETE /realm-journey/team/:teamId/:realm`) can let the team re-enter from the start.

---

### 1.3 Hints

**Endpoints [V]**

| Endpoint | Method | Notes |
|---|---|---|
| `GET /hint/task-overview/:taskId` | GET | `{unlocked:[Hint], locked:[Hint]}`. Requires task access (403 while locked). |
| `GET /hint/locked/:taskId` | GET | Locked hints for the task. |
| `GET /hint/unlocked/:taskId` | GET | Team's unlocked hints. **Returns `[]` even while tasks are locked** — this one is not task-access gated. |
| `POST /hint/unlock/:hintId` | POST | No body; `:hintId` is the hint's `id`, **not** the task id. |
| `GET /hint?taskId=&search=&limit=&offset=&order=DESC&orderBy=createdAt` | GET | Admin list. |
| `POST /hint`, `PATCH /hint/:id`, `DELETE /hint/:id`, `POST /hint/json` | — | Admin CRUD/import. |

**Hint object [V]**: `{id, taskId, textEn, textRo, penalty, createdAt}`. Admin form field label is exactly **"Penalty Points"**.

**Cost and behaviour [V]**

- Warning dialog before unlock: *"Be careful!"* + *"You lose {{penalty}} points by using the hint."*
- Locked-hint tooltip: *"This hint is currently unavailable. Unlocking it in the future will reveal the content and deduct points from your score."*
- Unlocked hints carry a permanent warning: *"Do not share this with other teams!"*
- **Team-level, per-task, irreversible.** No cooldown, no per-user limit, no refund endpoint visible. Multiple hints per task are allowed; penalties are additive `[I]`.
- Hints cannot be unlocked after the task time expires: `realm.hintTimeExpired = "Time is up for this task, hints can no longer be unlocked."` Server rejects with 403 (`taskTimeExpired` path). In Default (untimed) this should never trigger.
- Unlocking a hint after a successful submission may or may not retroactively reduce the already-counting score; the submission row stores its own `penalty`/`finalScore`. **Do not rely on a free-hint loophole**; treat the penalty as immediately charged to the task's score. **[open]**

**Strategy:** hints are priced in absolute points. Compare `penalty` to the expected gain and to `maxScore`; a hint on a 10-point task can zero it. `GET /hint/task-overview/:id` reveals the penalty schedule **without unlocking**, so collect it for every task at start.

---

### 1.4 Clue / passphrase and the Antarctica (Frozen) realm

**Live payload [V]** — `GET /clue/passphrase`:

```json
{
  "slots": [
    {"slot":1,"word":null,"taskId":"task_argentina","countryCode":"ar","unlockedAt":null,"trigger":null},
    ...
    {"slot":10,"word":null,"taskId":"task_united_kingdom","countryCode":"gb","unlockedAt":null,"trigger":null}
  ],
  "complete": false,
  "unlockedCount": 0,
  "total": 10
}
```

Exact 10 slot tasks: **Argentina, Australia, Belgium, Chile, France, Japan, New Zealand, Norway, South African Republic, United Kingdom**. `word`, `unlockedAt`, `trigger` become non-null on unlock. Unlocked words have no expiry in the payload; only an admin revoke/reset removes them. `clue:unlocked` socket events fire on each word (`{word, complete}`), so polling `/clue/passphrase` every few seconds or subscribing is enough to track progress.

**Clue trigger rule [V, admin text]**

> *"N slots · clue after pass or &lt;attemptLimit&gt; rejects · lock-after-attempts: on/off (submissions always allowed)"*

- A word unlocks either when the slot's task **passes** (`trigger:"success"`) or after **`attemptLimit` rejected submissions** (`trigger` likely `"attempts"`; admin can also grant manually as `trigger:"admin"`).
- `lockAfterAttempts` **on** means after the attempt limit the task stops accepting submissions — error **403 `clueLocked`**: *"No attempts left on this task — your clue is already unlocked."*
- `lockAfterAttempts` **off** means submissions remain open after the limit; the clue is still granted. The parenthetical *"(submissions always allowed)"* is rendered by the admin UI but conflicts with `clueLocked`; read it as "when off, submissions stay allowed".
- **Values of `attemptLimit` and `lockAfterAttempts` are admin-only and not exposed to participants.** No remaining-attempts counter is shown in the task UI. **[open]**
- Consequence: on a clue-linked task you can deliberately trade the task's points for the word by burning rejects; if locking is on, you lose the ability to solve that task afterwards. Decide per task before submitting.

**Passphrase bar [V]**

- Hidden entirely until at least one word is unlocked (`unlockedCount === 0 → null`).
- Shows 10 slots: `???` = locked; `•••` = unlocked but not yet revealed; word = revealed. Progress bar = `unlocked/total`.
- New word animation: a penguin delivers the word; toast `"Penguin brought a clue: {{word}}"`.
- On all 10: `complete=true`, snowfall overlay and toast **"Passphrase complete — Antarctica is open!"**.
- Admin monitor counts teams as **`Passphrases complete` → "eligible for Frozen"**, separately from **`In Frozen realm` → "entered Antarctica journey"**. Completion alone does **not** put a team in Frozen. **[V]**

**Realm mechanics [V]** (`/admin/info`, participant-visible only to admins but authoritative)

> *"A realm is the state a team is currently in. It decides which tasks the team sees on the map and the tasks page, and which ones it can submit and unlock hints for. Tasks of other realms are hidden and their submissions are rejected."*
> *"Every team starts in the Default realm. The other realm is Frozen: while a team is in it, the map shows its countries in white and a snowflake badge appears on the map, tasks and team pages."*
> *"A task can be a gateway to a realm ('Gateway to realm' in the task form). When a team member submits a gateway task and the submission passes, the whole team enters that realm."*
> *"A team never enters a realm whose tasks it has already all passed, so passing a gateway again does nothing once that realm is completed. The gateway is never shown to participants."*
> *"Scoring does not depend on realms: points from every realm add up to the team's score and the submission history covers all realms."*

**Frozen chain [V/I]**

- Admin empty-state: *"No tasks in the frozen realm yet. The **5-stage chain** (PR #271) must be seeded with `realm=frozen` before teams can enter after completing the passphrase."* — so Frozen is a **5-task chain**, not 24 tasks.
- Each stage is a normal task with `realm:"frozen"`, a `realmOrder`, and `timeLimitMinutes`; stages unlock in order, one at a time.
- All Frozen stages appear to be set in Antarctica (`countryCode:"aq"`); the map has a special `aq` layer and fills Antarctica green by `passed/total` while the team is Frozen **[I, from map chunk]**.
- Completion condition: latest submission of all 5 passed → team automatically returns to Default; realm marked completed. If the journey ends by timeout, the team can only leave (permanent).
- Admin can `grant`/`revoke`/`reset` clues and `DELETE /realm-journey/team/:teamId/:realm` to reset a team's Frozen journey ("they will be able to enter it again from the start").

**Gateway identity is not exposed.** No participant-visible UI submits the passphrase; the bar only displays words. The most likely design is: when `complete=true`, a hidden `task_antarctica` (country `aq`, `gatewayForRealm:"frozen"`) becomes available and its flag is the passphrase; passing it moves the team into Frozen. It is not in `GET /task/countries` now, consistent with "the gateway is never shown to participants". **Verify at start whether a new task/route appears at 10/10.** **[open]**

---

### 1.5 Submission rules

**Input types [V]**: `text` (`"Text (flag)"`), `file` (`"File upload"`), `prompt` (`"Prompt (LLM chat)"`). One per task.

**Text / flag [V]**

- `POST /submission/text` body `{taskId, text, url?}` (`url` omitted/null if empty).
- Required: non-empty flag (`submission.missing-flag = "Please provide a flag."`). No client max length.
- Optional `url` ("Link" field), validated as a URL (`validation.urlInvalid`).

**File [V]**

1. `POST /submission/upload-file-url` body `{taskId, fileSize, fileName, fileExtension, contentType}` → `{allowed, url, key, …}`.
2. `PUT <presigned url>` with raw bytes, `Content-Type` from the browser (no Bearer header).
3. `POST /submission` body `{taskId, fileStorageKey:key, url?}`.

Limits/validation:
- Client checks: extension present; extension in the task's accepted list; `fileSize <= maxUploadedFileSize`.
- Defaults: **max 10 MB**, admin can set **0.01–2047.99 MB** ("Defaults to 10 MB. Up to about 2 GB.").
- `acceptedFileExtensions` default **`*`**; comma/semicolon list, dots stripped, lowercase, `[a-z0-9]+` only.
- Exact errors: `submission.missing-file`, `submission.missing-file-extension`, `submission.fileExtensionNotAcceptedWithList`, `submission.fileTooLargeWithLimit`.
- Presigned upload failures: `errors.uploadFailed`, `errors.uploadRejected` ("The upload was rejected or took too long. Select the file again and resubmit.").
- Upload URL request can be throttled: `errors.tooManyRequests = "Please wait before submitting another file for this task."` (429).

**Prompt / LLM chat [V]**

- Same endpoint `POST /submission/text`; `text` is the prompt.
- Chat UI shows the team's last **20** submissions for the task (`Pc=20`), each with author/time, prompt bubble, and system reply from `submission.details`.
- Max prompt length **15,000 characters** (counter `n / 15000`); `Ctrl/⌘+Enter` sends.
- Critical semantics: *"No attempts yet. Write a prompt below. Each prompt is evaluated on its own, earlier attempts are not remembered."* and *"every prompt is evaluated on its own"*.
- Pending attempt renders *"Evaluating…"*; the evaluator reply is `details`; it can be `evaluationRefused` (with a message) or `evaluationFailed`.
- Because only the latest submission counts, **each prompt replaces the previous one for scoring**, even though the chat displays history.

**Status lifecycle [V]**

`pending → passed | rejected | error` (`Y.Pending/Passed/Rejected/Error`).

- `pending`: `"Submission received. Evaluation may take a few minutes, you will be notified when it is done."`; a team may not submit that task again → **409 `submissionPending`** (`"A submission for this task is still being evaluated…"`). `GET /submission/team/my/pending` returns pending task ids.
- `passed`: *"Submission passed ({{score}} pts)"*.
- `rejected`: *"Submission not passed ({{score}} pts)"*; may still carry partial `finalScore`.
- `error`: *"The submission could not be evaluated."*
- Live result via socket `submission:evaluated` with `{taskId,countryCode,status,finalScore}`; also pollable via `GET /submission/team/my`.

**Other submission errors [V]**

| HTTP/code | Meaning |
|---|---|
| 403 `taskTimeExpired` | Deadline passed (Frozen). |
| 403 `taskNotUnlocked` | Sequential task not yet unlocked. |
| 403 `realmLeft` | Team left the realm. |
| 403 `taskNotInRealm` | Task belongs to another realm (`"This task is not available in your current realm."`). |
| 403 `clueLocked` | Attempt limit exhausted on a clue task. |
| 403 `submissionsClosed` | Global `allowSubmissions=false`. |
| 403 `submissionFileMissing` | Upload key/file not found. |
| 409 `submissionPending` | Previous evaluation still running. |
| 429 `submissionThrottled` | *"Please wait a moment before submitting again."* |
| `evaluatorUnavailable` | Evaluation service down; *"Your submission was not counted, please try again…"* |
| `evaluationRefused` / `evaluationFailed` | Evaluator rejected/failed the content (message shown). |

**Practical concurrency rule:** one in-flight submission per task **for the whole team**; a pending evaluation blocks every member. Serialize submissions per task.

**GitHub URL**

- `ghUrl` is a **task metadata field** ("GitHub task URL", placeholder `https://github.com/…`, nullable) in the admin form — not a submission field. The submission field is the optional `url` ("Link").
- `ghUrl` is not rendered in the participant task panel in this bundle; it may be used by task authors inside the Markdown description. **[V/I]**

---

### 1.6 Fair play / anti-cheat

**The only platform-wide rule text is the Terms & Conditions [V]**, accepted at registration. Relevant exact clauses:

> *"Obtaining flags, scores, solutions, or other competition results through unauthorized methods, including cheating, unauthorized hacking, or interference with other teams;"*
> *"Accessing, stealing, sharing, or publishing other teams' solutions, flags, or confidential competition materials without authorization;"*
> *"Attempting unauthorized access to the platform, its infrastructure, accounts, databases, or other systems;"*
> *"Exploiting vulnerabilities in the platform or competition infrastructure outside the scope expressly authorized by the organizers;"*
> *"Interfering with the operation, availability, or security of the platform;"*

Disciplinary measures include warning, access restriction, removal of submissions, account suspension, and **"Disqualifying an individual participant or team"** (§9).

**Hint sharing [V]**: every unlocked hint is displayed with *"Do not share this with other teams!"*.

**No global model/API/data ban [V]**: the SPA bundle, map chunk, and full i18n catalogue contain **zero** occurrences of `pretrained`, `pre-trained`, `external API`, `plagiarism`, `fair play`, or `dataset` (except a Privacy Notice mention of task datasets sent to Modal). Therefore:

- Restrictions on pretrained models, external datasets, external APIs, LLM use, etc. are **per-task**, presumably inside the task's Markdown description.
- Apply the repo default-safe rule (no pretrained weights, no external datasets, no external APIs) until the task statement explicitly permits them. Note the prompt/LLM task type means *calling the platform's own evaluator* is allowed by design, not that external LLM APIs are allowed.
- Team-internal sharing is allowed (same team); cross-team sharing of flags/solutions/hints is explicitly prohibited and can lead to disqualification.

---

### 1.7 Event schedule, countdown and server time

**Site metadata [V]**

- `<title>`: `LMML 2026 - 24-Hour Machine Learning Marathon | Sigmoid`
- meta description: *"Participate in LMML 2.0, a 24-hour AI marathon with team challenges in NLP, Computer Vision, Reinforcement Learning, Data Science, and Generative AI."*
- Home page H1: **"Be ready to start your 24h Journey"** (`readyToStartJourney`: `"Be ready to start your <br /> <1>24h</1> Journey"`). No dates on the page.

**JSON-LD Event block in `index.html` [V]**

```json
{
  "@type": "Event",
  "name": "LMML 2026 - Learning Marathon of Machine Learning",
  "startDate": "2026-11-14T00:00:00+03:00",
  "endDate": "2026-11-15T23:59:00+03:00",
  "location": {"name": "Clasa Viitorului", "addressLocality": "Chişinǎu", "addressRegion": "Centru", "addressCountry": "MD"},
  "organizer": {"name": "Sigmoid", "url": "https://sigmoid.academy"}
}
```

Caveats: the span is ~48h, not 24h; the offset `+03:00` does not match Chisinau's `+02:00` on 14 Nov (EET). Treat these dates as **approximate/marketing metadata**, not the authoritative start trigger.

**Live status at 2026-11-14 08:54 UTC / 10:54 EET [V]**

- `GET /settings` → `{"allowSignUp":true,"allowSubmissions":false}` (optional date fields absent).
- `GET /realm-journey/my` → `{"serverNow":"2026-11-14T08:54:05.093Z","journey":null}`.
- `GET /task` → 403 `task access is denied`; hints → 403.
- So the event had **not** started at 10:54 local, despite the JSON-LD start date. Expect the organizers to toggle `allowSubmissions` (and possibly set the date fields) manually at the opening ceremony.

**How opening/closing actually works [V]**

- `settings.allowSubmissionsInfo`: *"Regular users can open tasks, submit solutions and unlock hints. When off, they are blocked from all this. Admins are not affected."*
- `submissionsWillOpenAtInfo`: *"Shown to participants only, as the start of the header countdown. It does not open submissions by itself, turn on 'Allow to make submissions' for that."*
- `submissionsWillCloseAtInfo`: *"Shown to participants as the header countdown deadline, the countdown is hidden without it. It does not close submissions by itself, turn off 'Allow to make submissions' for that."*
- Header countdown appears only when `submissionsWillCloseAt` is set (`Ou` timer; label `timeRemainingUntilClose`, tooltip `Deadline: {{date}}`).
- When `allowSubmissions` flips false, regular users lose access to tasks/hints/submissions (`canAccess = isAdmin || settings.allowSubmissions`). Existing scores remain.
- Socket events: `settings:updated` (with `submissionsOpened/submissionsClosed`), `appPhase:update` (phase strings `registration|active|closed`), `leaderboard:update`.

**Server time [V]**

- `serverNow` is returned by `GET /realm-journey/my`. The SPA corrects local clock skew with it and uses it for all deadlines; do the same rather than trusting the workstation clock.
- We observed host clock and server clock matching within ~1 s.

---

### 1.8 Task set, country map, complexity, "24 levels"

**Task addressing [V]**

- The SPA addresses a task by **country code**: `GET /task/:countryCode` (`findTask("ar")`), not by `task_argentina`. It also accepts the task id by convention **[I]**.
- List: `GET /task?limit=100&offset=0` → `{items:[Task], totalItems}`.
- `GET /task/countries` → array of country codes with tasks (used to enable map regions). **Currently 54 codes**:

```
eg ee et fj al ao ar au be bt bo br cl cn co cr ci cu ec fi fr de gr is in jp kz la
mx md mn ma nz ng no pa pg pe pt rw ws sn za es ch tz gb us ve vn ca dz ro mz
```

Distribution: Europe 15, Africa 12, South America 8, Asia 8, North America 6, Oceania 5, Antarctica 0 (not seeded/visible). This may include historical/internal tasks; the authoritative count is `/task.totalItems` at start. **Do not assume 24.** **[V/I]**

**Continent = domain [V]** (from `md[...]` in the bundle + `continent.*` i18n):

| Continent | Domain label |
|---|---|
| Europe | **Classic ML & Data Science** |
| Asia | **Computer Vision** |
| Africa | **General AI** |
| North America | **NLP** |
| South America | **AI Security & Adversarial ML** |
| Oceania | **Reinforcement Learning** |
| Antarctica | **Special Challenge** (Frozen chain) |

**Task list / map [V]**

- Tasks page groups by continent in order `europe, asia, africa, northAmerica, southAmerica, oceania, antarctica, other`.
- Within a continent: sort by `realmOrder` (null last), then localized country name.
- Map: countries with tasks are clickable and colored by continent; countries without tasks are disabled/grey. Team progress colors a country green (`passed:true`) or red (`passed:false`) via `GET /team/my/country-progress` → `[{countryCode, passed}]`.
- In Frozen: white/ice theme, snowflake badge, Antarctica fill = `passed/total` stages.
- `complexity` is free text but the UI colors exactly `easy` (lime), `medium` (orange), `hard` (red); anything else muted. Expect these three labels.
- Task detail shows `Country`, title, Markdown description, `Complexity: <value>`, hints, submission form. Before start it shows blurred lorem ipsum and *"Task details are hidden until you start."*

**Profile / team gates [V]**

- Tasks are locked until participant info is complete: *"Tasks are locked until your participant information is complete."* Required: first name, last name, phone, age, gender, residence environment, location type, and (capital or region name). Our profile is complete.
- Must be in a team (max 5) to submit; `"You should be in a team"`.
- `allowSubmissions=false` blocks task pages/hints/submissions for non-admins regardless of profile.

---

## 2. Inferred behavior (must be confirmed live)

| # | Inference | Confidence | How to confirm |
|---|---|---|---|
| 1 | `finalScore = max(0, evaluatedScore − Σ hint penalties)`; task score is `finalScore`, capped at `maxScore` (or not) | High on subtraction, low on cap | First task with a hint: compare `evaluatedScore`, `penalty`, `finalScore` in `GET /submission/team/my`. |
| 2 | Team score = sum of per-task counting `finalScore`; member `scoreContributed` = that member's counting submission scores | High | Compare `GET /scoring/team/37` with sum of history after 2–3 submissions. |
| 3 | Latest submission per task replaces the previous counting one, even if worse/rejected | High (explicit admin text) | Low-stakes task: submit known-bad, then known-good, then a worse one; watch team score after each. |
| 4 | Passing a task unlocks the next Frozen stage; failing its deadline ends the journey | Medium-high | Observe first Frozen stage; do not gamble the chain before confirming. |
| 5 | Default-realm tasks have no per-task deadline; `timeLimitMinutes` only applies to Frozen | High (client skips journey unless Frozen) | Check task page for timer after start; if no timer and `journey=null`, confirmed. |
| 6 | Completing the 10-word passphrase exposes a hidden `task_antarctica` gateway; passing it enters Frozen | Medium | At 10/10, re-fetch `/task`, `/task/countries`, `/task/aq`; inspect map/task list. |
| 7 | Unlocking a hint after a passing submission retroactively reduces that task's score | Low | Compare team score before/after unlocking a hint on an already-passed low-value task (only if safe). |
| 8 | 429 throttle is a short per-task cooldown (seconds), not a daily quota | Medium | Observe `Retry-After`/repeat timing on first 429. |
| 9 | Tie-break is server-side by score desc, then some hidden recency/first-to-score key; ranks are always sequential | Medium | Watch two teams with equal non-zero scores; client order is authoritative. |
| 10 | Internal `(Sigmoid team)` entries are excluded from prizes but included in the public table | Low | Organizer announcement; not in platform text. |

---

## 3. Open questions

1. **Exact final-score formula and cap.** Is `finalScore = evaluatedScore − penalty`, and is it clamped to `[0, maxScore]`? Are hint penalties charged once per hint, per submission, or recalculated on every read?
2. **Latest vs best.** The UI says latest, but does the server actually let a worse submission overwrite a better one? Does a passed task accept further submissions? (High priority: one cheap experiment.)
3. **Attempt limit value and `lockAfterAttempts` state.** Not exposed to participants. Is it 3? 5? Does a *pass* count as an attempt, or only rejects?
4. **Frozen chain seeding.** Are the 5 stages actually seeded? Which task IDs/country codes? What is the first stage's start trigger? Does stage timeout end the whole journey or unlock the next stage?
5. **Gateway task identity.** Is it `task_antarctica` / country `aq`? How is it exposed at 10/10? Is passing it automatic or a normal text submission?
6. **Default per-task timers.** Does the server enforce `timeLimitMinutes` on Default tasks even though the client doesn't show a journey timer?
7. **Tie-break rule.** Score, then what? Earliest time to reach score? Last submission time? Random/cached order?
8. **429 thresholds.** Window and penalty for text vs file upload; is there a per-task or global request budget?
9. **Actual task count and max scores.** 54 country codes now vs the "24" branding. `GET /task.totalItems` and `Σ maxScore` at start.
10. **Historical tasks.** Are all 54 `/task/countries` entries live competition tasks, or do some belong to earlier/internal test events?
11. **Hint retroactivity.** Can a team unlock a hint after passing and avoid the penalty? (Probably not; verify defensively.)
12. **Prompt evaluator safety.** What triggers `evaluationRefused`? Does a refused prompt count as an attempt/reject for clue purposes?

---

## 4. Strategic implications

**Scoring / submission discipline**

- **One writer per task.** Because the latest team submission counts, two teammates submitting to the same task can accidentally overwrite a good result. Maintain a claim table (`task_id → owner`) and a "do not resubmit after pass" rule unless the owner has a verified improvement.
- **Never submit a placeholder to "test the API" on a task you have already passed.** Use one low-value task for the first scoring-semantics experiment, then freeze it.
- **Pending blocks the whole team.** Wait for `submission:evaluated` or `GET /submission/team/my/pending` to clear before the next attempt on that task. 409 is a team-wide lock, not per-user.
- **Partial credit is real.** A rejected file/prompt submission can still add points; never assume a reject is zero. Conversely, a later reject can lower the total.

**Hints / clues**

- Pull `GET /hint/task-overview/:id` for every task at start; it reveals penalties without spending anything. Rank tasks by `maxScore / (penalty of needed hints)`.
- A hint on a low-value task can wipe its entire value. Prefer solving, then hints only on high-value blockers.
- For the 10 clue tasks, attempt them **before** burning hints: a pass gives the word *and* points. If stuck, remember that `attemptLimit` rejects also gives the word — but may permanently lock the task. Decide whether the word or the task's points matter more.
- The passphrase bar is public to the team; no input is needed. Do not waste time looking for a passphrase submission box until 10/10; at that point re-scan the task list/map for a new Antarctica entry.

**Frozen / Antarctica**

- Do not enter Frozen casually. It is a **5-stage, sequential, per-stage-timed** chain. Prepare the full toolchain for all five stages *before* passing the gateway; there is no pause, and leaving is permanent.
- Assign one dedicated operator to the chain and watch the 10-minute red timer. A single missed deadline can end the journey and forfeit the remaining stages.
- Complete the 10 clues opportunistically during the main 24h; they are a bonus realm, not the main score. Scoring is realm-independent, so Frozen points add on top of Default points.
- If the chain is buggy/unseeded (possible: admin empty-state + PR #271 reference), do not sacrifice Default points chasing it. Verify with admins.

**Triage / parallelism**

- If ~54 tasks are live across 7 domains, 24h gives ~25 min/task average for a team. Use the map: easy/medium first, hard later. Parallelize by continent/domain across members.
- Europe = Classic ML/DS, Asia = CV, North America = NLP, South America = Security/Adversarial, Oceania = RL, Africa = General AI. Match members to strengths; the domain is visible before opening a task.
- Flag/text tasks are the fastest wins; file tasks need upload + evaluation wait; prompt tasks need iterative chat but each attempt is independent.

**Fair play / risk**

- Read each task's Markdown description for dataset/pretrained/API restrictions *before* coding. The platform itself does not enforce a global ban, but ToS §9 allows disqualification for cheating or cross-team sharing.
- Never share flags, hints, weights, or solutions outside HardCoders. Hint UI explicitly warns against it.
- Do not probe admin endpoints, other teams' submissions, or platform vulnerabilities. ToS explicitly lists unauthorized access and interference as disqualifying.

**Operational**

- Sync to `serverNow`, not the laptop clock. Deadlines are server-enforced; a local clock skew of seconds can cost a submission.
- Keep a poller on `/settings` (open/close), `/realm-journey/my` (Frozen journey/deadlines), `/clue/passphrase` (words), `/submission/team/my/pending` (evaluation locks), and `/scoring/leaderboard` (10 s).

---

## 5. "What to do at start" checklist

Run in this order as soon as `GET /settings.allowSubmissions` becomes `true` (or `/task` stops returning 403):

1. **Snapshot state (no writes):**
   ```bash
   cd /home/kozak/mllm/MML
   PY=/home/kozak/mllm/py
   $PY tools/sigmoid_api.py settings
   $PY tools/sigmoid_api.py journey
   $PY tools/sigmoid_api.py tasks --params '{"limit":100,"offset":0}'
   $PY tools/sigmoid_api.py clue
   $PY tools/sigmoid_api.py submissions
   ```
2. **Pull the full task inventory** with `tools/intake.py` (or `GET /task?limit=100`) and record per task: `id`, `countryCode`, `maxScore`, `complexity`, `inputType`, `timeLimitMinutes`, `realm`, `realmOrder`, `gatewayForRealm`, `maxUploadedFileSize`, `acceptedFileExtensions`, `ghUrl`, description links.
3. **Fetch every hint penalty schedule** with `GET /hint/task-overview/:taskId`. Do **not** unlock. Build `penalty per hint` vs `maxScore` table.
4. **Read every task description's rules section** for pretrained/API/dataset restrictions before assigning work. Default-safe until explicitly allowed.
5. **Check the actual task count** (`totalItems`) and continent/domain split. If it is ~54, immediately triage into easy/medium/hard and assign owners. Do not assume 24.
6. **Run one scoring-semantics experiment** on the cheapest task (prefer text, low `maxScore`, not a clue task):
   - submit a deliberately bad answer; record `status`, `evaluatedScore`, `penalty`, `finalScore`, team `score`;
   - submit a correct answer; record the same;
   - (only if safe) submit a worse answer; check whether team score drops.
   This resolves latest-vs-best, partial credit, and the formula in one pass.
7. **Set the team submission protocol:** one owner per task; no resubmit after pass unless a verified improvement; wait for `pending` to clear; never submit while another member is mid-upload.
8. **For file tasks:** pre-check extension and size (default 10 MB, per-task overrides); upload promptly; if 429, back off and retry the whole upload-URL → PUT → POST flow.
9. **For prompt tasks:** remember each prompt is independent, max 15,000 chars, last 20 shown; the latest prompt is the only one that counts. Keep a prompt log per task.
10. **For the 10 clue tasks:** log attempts and reject counts; decide per task whether to spend the attempt limit for the word or stop one attempt short to keep the task solvable (if locking is on). Re-check `GET /clue/passphrase` after every pass/reject.
11. **At 10/10:** re-fetch `/task`, `/task/countries`, `/task/aq`, `/realm-journey/my`; look for the Antarctica gateway. Do not pass the gateway until the 5-stage toolchain is ready and a dedicated operator is free.
12. **If in Frozen:** watch `deadlineAt` via `/realm-journey/my` + `serverNow`; treat <10 min as red alert; pass each stage before expiry; do not leave unless the journey has ended or the chain is confirmed broken (leave is permanent).
13. **Monitor** `/settings` (close), `/scoring/leaderboard`, `/submission/team/my/pending`, and socket events (`settings:updated`, `submission:evaluated`, `clue:unlocked`, `team:realmChanged`, `team:realmJourneyUpdated`).
14. **Fair play check** before any external download/API call: task description explicit permission? If not, no pretrained models, no external datasets, no external APIs. Never share anything cross-team.

---

## Appendix A — Mechanics-relevant endpoint quick reference

| Area | Endpoint | Method | Notes |
|---|---|---|---|
| Settings | `/settings` | GET/POST | public GET; POST admin |
| Tasks | `/task?limit=&offset=` | GET | list, 403 until start |
| Tasks | `/task/:countryCode` | GET | SPA uses country code |
| Tasks | `/task/countries` | GET | enabled map countries |
| Team progress | `/team/my/country-progress` | GET | `[{countryCode,passed}]` |
| Journey | `/realm-journey/my` | GET | `{serverNow,journey}` |
| Journey | `/realm-journey/my/leave` | POST | body `{teamName}` |
| Journey | `/realm-journey/team/:teamId/:realm` | DELETE | admin reset |
| Hints | `/hint/task-overview/:taskId` | GET | locked/unlocked + penalties |
| Hints | `/hint/locked/:taskId`, `/hint/unlocked/:taskId` | GET | lists |
| Hints | `/hint/unlock/:hintId` | POST | irreversible, charges penalty |
| Clues | `/clue/passphrase` | GET | 10 slots, words, complete |
| Clues | `/clue/admin/overview` | GET | admin: passphrase, attemptLimit, lock flag, teams |
| Clues | `/clue/admin/teams/:teamId/grant|revoke|reset` | POST | admin |
| Submit | `/submission/text` | POST | text & prompt |
| Submit | `/submission/upload-file-url` | POST | presigned upload |
| Submit | `/submission` | POST | finalize file |
| Submit | `/submission/team/my?limit=&offset=&taskId=` | GET | history |
| Submit | `/submission/team/my/pending` | GET | pending task ids |
| Score | `/scoring/leaderboard` | GET | team ranking |
| Score | `/scoring/team/:teamId` | GET | `{score}` |
| Score | `/scoring/leaderboard/:teamId/details` | GET | counts + per-member contribution |

## Appendix B — Exact string pack (for copy/paste into UI checks)

- `"Be ready to start your <br /> <1>24h</1> Journey"`
- `"Tasks of a timed realm unlock one at a time, lowest first. Empty goes last, by creation time."`
- `"How long the task takes submissions once unlocked. Changing it does not move deadlines already running."`
- `"Passing this task moves the team into this realm."`
- `"A realm only counts as completed when the team's latest submission of every one of its tasks passed."`
- `"The gateway is never shown to participants."`
- `"Scoring does not depend on realms: points from every realm add up to the team's score and the submission history covers all realms."`
- `"Checked: only the latest submission per task, the ones that add up to the team score."`
- `"You lose {{penalty}} points by using the hint."`
- `"This hint is currently unavailable. Unlocking it in the future will reveal the content and deduct points from your score."`
- `"Do not share this with other teams!"`
- `"Penguin brought a clue: {{word}}"`
- `"Passphrase complete — Antarctica is open!"`
- `"N slots · clue after pass or <attemptLimit> rejects · lock-after-attempts: on/off (submissions always allowed)"`
- `"No attempts left on this task — your clue is already unlocked."`
- `"Each prompt is evaluated on its own, earlier attempts are not remembered."`
- `"Your team can submit this task again once the evaluation is done."`
- `"Please wait a moment before submitting again."`
- `"Regular users can open tasks, submit solutions and unlock hints. When off, they are blocked from all this. Admins are not affected."`
- `"It does not open submissions by itself, turn on 'Allow to make submissions' for that."`
- `"It does not close submissions by itself, turn off 'Allow to make submissions' for that."`
- `"Disqualifying an individual participant or team."`
