# 04 — War-Room Operating Plan (AI Subagent Swarm)

**Version:** 1.0 · **Date:** 2026-11-14 · **Owner:** Subagent D
**Scope:** How the orchestrator spawns, feeds, and reaps AI subagents to solve 24 sequential,
server-timed LMML 2026 tasks on one 4-core / 15 GB CPU machine.

**Authority order when documents disagree:**
`task statement` > `reports/AGENT_CONTEXT.md` > this plan > repo `AGENTS.md`.
Repo `AGENTS.md` still describes a Windows/RTX/Modal machine. The live box is Linux,
has no GPU, and runs Python through `/home/kozak/mllm/py`. Trust probes, not prose.

**Contents:** 0 Ground truth · 1 Principles · 2 Roles · 3 Lifecycle · 4 Parallelism ·
5 Files/git · 6 Artifacts · 7 Escalation · 8 First 15 min · 9 Reporting ·
10 24-h rhythm · 11 Do-not-do · 12 Spawn prompts · 13 Done.

---

## 0. Ground Truth (re-verify at start; do not assume)

| Item | Value / Rule |
| :--- | :--- |
| Machine | 4 vCPU, ~15 GB RAM (~10 GB available), no GPU, ~78 GB disk |
| Repo | `/home/kozak/mllm/MML` (remote `origin`, branch `main`) |
| Python | **Always** `/home/kozak/mllm/py`; direct `venv/bin/python` may fail on NixOS libs |
| Task order | Sequential (mission brief: 24). **Confirm `GET /task.totalItems`** — rules recon saw ~54 country codes, so treat 24 as the planning assumption, not a fact. Each task may have its own `deadlineAt`; Default-realm tasks may only be bounded by the global submissions close. |
| Submission model | Text flag, file upload, or prompt. **Only the latest submission per task counts.** A pending evaluation blocks resubmission. Hints deduct points. **Rejected can still carry partial score**, but a later worse attempt can lower the total. |
| Clock | Server time is authoritative. A task that expires stops accepting submissions and hints. |
| Secrets | JWT at `/home/kozak/mllm/.sigmoid_token`; GitHub token at `/home/kozak/mllm/gittoken`. Both outside the repo. |
| Git | Subagents run **no git commands**. Orchestrator is the only committer/pusher. |
| Fair play | Default = **deny** pretrained models, external datasets, external APIs. Enable only when the task text explicitly allows it. |

**The most important consequence:** a subagent is an LLM context, not a machine.
Spawning five agents does not create five CPUs. The orchestrator must serialize compute
and use agents for thinking, I/O, verification, and writing — not for parallel training.

### 0.1 Quick Card (pin this)

- **Crew:** orchestrator + intake + 1–2 solvers + reviewer + submission officer + scribe; env fixer on call.
- **Max 4 subagents, 1 heavy job, 3 compute threads, 1 submission in flight.**
- **Cycle:** T+0 intake → T+5 EDA → T+15 baseline → T+35 improve → T+55 validate → T+70 submit → T+80 report → reserve.
- **Bank a valid submission on every task; latest submission counts.**
- **No subagent git. Orchestrator commits/pushes; subagents write their sandbox + report only.**
- **Default-deny fair play; quote the task before enabling anything.**
- **Escalate P0 immediately: deadline, 401, locked task, missing data, metric/fair-play ambiguity.**
- **Report format:** `STATUS <role> <task> | DONE | DOING | BLOCKED | ETA | FILES`.
- **Done:** accepted submission + ledger + validation + report + backup + closeout.

---

## 1. Operating Principles

1. **Bank first, improve second.** A valid submission on every task beats a perfect model on half the tasks.
2. **The server clock wins.** Freeze code at deadline − max(5 min, 10% of remaining). Never start an upload that cannot finish.
3. **One owner per artifact, one writer per file.** No shared editing, no two agents in one run directory.
4. **Evidence over claims.** A result is real only with command, exit code, metric value, and file hash.
5. **Latest submission counts.** Never upload a candidate unless it beats the current counted submission or is the known-good fallback. Rejected ≠ zero: partial credit exists, but a later worse attempt can still lower the total.
6. **Default-deny fair play.** Explicit permission only; quote the exact task line in the brief.
7. **CPU discipline.** Total compute threads ≤ 3; one heavy job at a time; check free RAM before every spawn.
8. **No speculative work on locked tasks.** Prepare only public/recon material until a task unlocks.
9. **Independent review.** The solver never signs off its own submission.
10. **Every task ends clean:** ledger entry, backup copy, final report, closeout, orchestrator commit.
11. **Blocked > 15 min = switch.** Fall back or move to the next task; do not burn a window on one error.
12. **No secrets, raw data, weights; reports ≤ 20 MB, repo hard stop 50 MB. No cross-team sharing.**

---

## 2. Role Catalog

### 2.1 Role Table

| Role | Count | Spawn when | Owns | Exit / kill |
| :--- | :--- | :--- | :--- | :--- |
| **Orchestrator** | 1 (main session) | Always | Clock, queue, spawn/reap, GO/NO-GO, `lvl_XX` promotion, git, final status board | End of event |
| **Intake / recon** | 1 per task | T−2 min before unlock, or T+0 | Task JSON, rules extract, fair-play matrix, data manifest, EDA/CTF scan, `TASK_BRIEF` | After brief approved + baseline command handed off (T+15); may convert to analyst |
| **Track solver** | 1 default, 2 max | T+8 min (after brief v0) | `work/tasks/<id>/runs/<run>/` only: code, models, predictions, candidate submission, `RESULT.json` | After candidate frozen or deadline; 10 min no artifact = kill |
| **Reviewer / validator** | 1 per task | T+10 min (standby), activates T+50 | `validation/`, `reports/levels/<id>_VALIDATION.md`; reproduction, leakage audit, format gate | After PASS/FAIL sign-off; can review next task |
| **Submission officer** | 1 per 6 h shift | At journey start | `submissions/ledger.jsonl`, `reports/levels/<id>_SUBMISSIONS.md`, all API submit calls | End of shift; handoff with ledger |
| **Report scribe** | 1 per task or 1 persistent | T+60 min for ≥60-min tasks, or immediately after final submit; solver self-writes for ≤45-min tasks | `reports/levels/<id>.md`, `reports/levels/LESSONS.md` | After report committed by orchestrator |
| **Environment fixer** | 1 on-call | First environment failure | Env diagnosis, deps, `py` wrapper, Rust/Playwright/Modal, OOM mitigation | 15 min timebox; back to on-call |

**Hard cap: 4 live subagent sessions** (excluding the orchestrator). Typical steady state:
1 solver + 1 reviewer + 1 scribe + 1 submission officer. During the intake burst,
intake + 2 solvers + reviewer = 4; no scribe yet. The environment fixer counts toward the cap:
during an incident, pause the scribe (reports can be written after the fix).

### 2.2 Role Cards

**Orchestrator**
- Owns the task queue, the wall clock, and the GO/NO-GO decision.
- Spawns only with a filled `TASK_BRIEF.md` and an explicit stop condition.
- Reads every agent reply in the standard status format (§9.3); never accepts "it works" without an artifact path.
- Promotes a winning run to `lvl_XX/` and commits; never lets a solver write `lvl_XX/`.
- Emergency rule: if the submission officer is unavailable, the orchestrator may execute the API call, but must read the checklist aloud (two-key rule degenerates to one key only in the last 10 min).

**Intake / recon**
- First command: `tools/intake.py` (or `--task-id`) to fetch and normalize the task and attachments.
- Parses `inputType`, `maxScore`, `complexity`, `timeLimit`/`deadlineAt`, `maxUploadedFileSize`, `acceptedFileExtensions`, sample submission, metric, target/id columns, hint penalties.
- Runs EDA + CTF forensics **before** any modeling: `optimizations/fast_eda.py`, `optimizations/ctf_tactics.py`, `rust_core/target/release/lmml_rs eda`.
- Produces the brief in ≤ 10 min. Flags unknowns explicitly; does not guess metric or fair play.
- After handoff, may stay as a second pair of eyes for feature ideas, but must not touch the solver's run directory.

**Track solver**
- Works only inside its own `work/tasks/<id>/runs/<run_id>/` sandbox.
- Uses the track template as a starting point, not a religion. CTF/flag tasks usually need custom code.
- Saves every experiment: `run.log`, `config.json`, `RESULT.json`, `submissions/submission.csv`, `oof.npy`, `test.npy`.
- Reports progress every 10 min in the status format; asks for a decision instead of silently changing the metric.
- Never submits, never edits `lvl_XX/`, never runs git.

**Reviewer / validator**
- Must not be the solver. Reproduces the candidate from the solver's command in a clean shell.
- Checks: split integrity, target leakage, train/test contamination, metric direction, ID alignment, row count, NaN/Inf, constant predictions, probability range, class mapping.
- Runs `tools/validate_submission.py` (preferred) and/or `lmml_master.py check` or the equivalent non-CSV validator.
- Signs `PASS` / `FAIL` with reasons and evidence. A FAIL sends the solver back with a specific defect, not a vague complaint.
- If CV looks implausibly high (> 0.99 on a noisy task), treat as leakage until proven otherwise.

**Submission officer**
- The only role allowed to call `POST /submission*` (orchestrator may override in an emergency).
- Runs `SUBMISSION_CHECKLIST.md` for every candidate and records every attempt.
- Checks for pending submissions before posting; backs off on `submissionPending` / `submissionThrottled`.
- Records API submission id, status, score, feedback, and the exact file hash.
- Never edits model code or reports other than the submissions ledger.

**Report scribe**
- Writes `reports/levels/<id>.md` from artifacts, not memory.
- Includes score, submission id, CV, validation verdict, fair-play decision, commands, and lessons.
- Keeps the lessons file current; flags template improvements to the orchestrator.
- Must finish within 10 min of the final submission; the report is part of "done".

**Environment fixer**
- Invoked on: import/lib errors, `py` wrapper failures, Rust build issues, Playwright/node issues, OOM, disk pressure, Modal auth.
- Diagnoses with exact commands, applies the smallest fix, and writes a 5-line incident note.
- Does not refactor task code or change templates without orchestrator approval.
- Hard timebox 15 min; if unresolved, recommends a fallback (pure NumPy, `LMML_NO_RUST=1`, smaller data, skip).

### 2.3 Track Solver Matrix

| Track | Template | First move on CPU | Playbook | Avoid first |
| :--- | :--- | :--- | :--- | :--- |
| NLP | `templates/01_nlp_fast.py` | `--mode tfidf` + linear/LGBM; char+word n-grams | `04_pro_nlp_and_llm_hacks.md` | Transformer training before a TF-IDF baseline |
| Tabular | `02_tabular_fast.py` | `--models lgb,ridge`, 5-fold, early stopping | `02_pro_feature_engineering.md` | Optuna before a valid submission |
| CV | `03_cv_fast.py` | `--mode extract_probe` (frozen features + linear head) if pretrained allowed; else small CNN from scratch | `03_pro_cv_and_vision_tricks.md` | Full fine-tune with no GPU and no baseline |
| Audio / GenAI | `04_audio_genai_fast.py` | `audio_clf` (Spec2D/ResNet1D) or `audio_forensics`; RAG only if corpus given | `05_pro_audio_and_genai_hacks.md` | Whisper/LLM fine-tune on CPU |
| Adversarial / Sec | `05_adversarial_sec.py` | `--mode forensics` first; then evasion/backdoor/poison per statement | `06_pro_adversarial_and_ctf_redteam.md` | Running all attack modes blindly |
| RL | `06_rl_fast.py` | PPO, `--n-envs 2`, short horizon, multi-seed if time | `07_pro_rl_and_control_hacks.md` | Huge sweeps or 16 parallel envs |
| Frontier | `07_frontier_gnn_pinn.py` | GNN on provided graph, PINN on provided PDE, QUBO on provided matrix | `08_pro_frontier_gnn_pinn_qml.md` | QML at scale; GPU assumptions |

**Prompt / LLM-chat tasks:** do not train a model. Craft one strong prompt from the task
context, submit via `POST /submission/text` (max 15,000 chars), and remember each prompt is
evaluated independently while only the latest counts.

**Template caveat:** templates emit `submissions/submission.csv` + `RESULT.json` for demo-style
tasks. Real tasks may require a flag, a ZIP, a JSONL, a prompt answer, or a custom file.
The solver adapts the output; the submission officer validates against the actual task rules.
For modality triage (CSV vs text vs image vs audio vs graph), use `reports/03_PLAYBOOK_CPU.md` §1.2.

### 2.4 Coordination Topology

- The orchestrator is the **hub**; subagents never talk to each other directly.
- All handoffs go through files: brief → run dir → validation → ledger → report.
- Files are durable memory; an agent's context is disposable and may be killed at any time.
- One shared task board and one submission ledger; no per-agent copies.
- Each subagent has exactly one owned path and one current assignment.
- If two agents disagree, the orchestrator decides from artifacts; the reviewer's PASS/FAIL is binding for submission unless the orchestrator records an explicit override.

---

## 3. Task Lifecycle and Timeboxes

### 3.1 State Machine

```
NEW ──► BRIEFED ──► BASELINE_RUNNING ──► BASELINE_SUBMITTED
                                              │
                                              ▼
                                         IMPROVING ──► VALIDATING ──► SUBMITTED ──► REPORTED ──► CLOSED
                                              │              │
                                              ▼              ▼
                                           BLOCKED        FAILED ──► IMPROVING (one retry)
```

Side states: `EXPIRED` (server deadline passed), `SKIPPED` (no viable path; report + close).
Every transition requires the artifact named in the gate below; a state without an artifact is not a state.

### 3.2 Two-Hour Task — Core 80-Minute Cycle + Reserve

| Clock | Phase | Owner | Hard output | Gate to proceed |
| :--- | :--- | :--- | :--- | :--- |
| **T+0–5** | Intake | Intake | `TASK_BRIEF.md` v0: metric, format, fair play, deadline, sample submission | G0: brief complete; deadline written in UTC |
| **T+5–15** | EDA / recon | Intake + solver | `manifest.json`, EDA/CTF notes, baseline command, CV plan | G1: data readable; sample parsed; metric + direction known |
| **T+15–35** | Baseline | Solver | Simplest valid candidate + `RESULT.json` | G2: `check` PASS; API accepts (pending OK); submission id recorded |
| **T+35–55** | Improvement | Solver | One better candidate (features/model/seed), OOF saved | G3: OOF beats baseline or explicit diversity reason |
| **T+55–70** | Validation / ensemble | Reviewer (+ solver) | `VALIDATION.md` PASS/FAIL, frozen candidate + hash | G4: reproduction PASS; no leakage; format PASS |
| **T+70–78** | Submit improved | Submission officer | API accepted, status recorded | G5: API accepted; ledger updated; score when available |
| **T+78–80** | Report v1 | Scribe | `TASK_REPORT.md` v1 | G6: report + ledger + backup exist |
| **T+80–100** | Reserve wave (optional) | Solver + reviewer | Second candidate only if clear hypothesis and ≥ 20 min left | G3/G4 again; otherwise keep current best |
| **T+100–105** | Final submit / confirm | Submission officer | Best file accepted; hash confirmed | G5: API accepted; ledger updated |
| **T+105–108** | Final report / closeout | Scribe + orchestrator | Final report, closeout JSON, commit | Task CLOSED |
| **T+108** | Hard stop (10% safety margin) | Orchestrator | No code/model/upload changes | — |
| **T+108–120** | Evaluation buffer / next task prep | Orchestrator | Pending resolved, next brief started | Next task |

**Rules for the reserve:**
- Only start wave 2 if (a) evaluation came back fast, (b) there is a concrete hypothesis, (c) ≥ 20 min remain before the final-submit window, and (d) the current best file is backed up and re-submittable.
- Never risk the counted submission for a marginal CV gain. If in doubt, stop at T+80 and use the reserve for report/next-task prep.
- If the task limit is actually 90 min, the core cycle ends at T+80 and the 10% safety margin ends at T+81. Do not start wave 2: confirm the counted submission/report by T+80 and use T+81–90 as evaluation buffer.

### 3.3 Adaptation to Shorter / Longer Limits

**Default assumption:** 24 tasks in 24 h ≈ **60 min/task**. Treat the 2 h cycle above as a
ceiling, not a target. If the next task unlocks automatically, its timer may start while the
previous report is still being written; start intake immediately and let the scribe finish in
parallel. A task that overruns eats the next task's window.

Percentages: intake 3–5%, EDA 8–12%, baseline 15–20%, improvement 30–40%,
validation 12–18%, submit 5–8%, report 5–7%, buffer 5–10%.

| Limit | Intake | EDA | Baseline | Improve | Validate | Submit | Report | Buffer |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **20 min** (CTF/flag) | 0–2 | 2–4 | 4–8 | 8–12 | 12–15 | 15–17 | 17–18 | 18–20 |
| **30 min** | 0–2 | 2–5 | 5–11 | 11–18 | 18–23 | 23–25 | 25–27 | 27–30 |
| **45 min** | 0–3 | 3–7 | 7–15 | 15–27 | 27–33 | 33–37 | 37–40 | 40–45 |
| **60 min** | 0–3 | 3–8 | 8–18 | 18–34 | 34–42 | 42–47 | 47–52 | 52–60 |
| **90 min** | 0–4 | 4–11 | 11–25 | 25–50 | 50–62 | 62–72 | 72–79 | 79–90 |
| **120 min** | 0–5 | 5–15 | 15–35 | 35–55 | 55–70 | 70–78 | 78–80 | 80–120 (optional wave 2; hard stop 108) |
| **180 min** | 0–6 | 6–18 | 18–42 | 42–105 (2 waves) | 105–130 | 130–148 | 148–160 | 160–180 |

- **≤ 20 min:** use the CTF/flag fast path, not the ML lifecycle. For 10–15 min, submit first and write the report immediately after the task closes.
- **≤ 45 min:** one solver; reviewer does a checklist pass in parallel; no separate scribe (solver writes the report).
- **60–90 min:** full roles, one improvement wave.
- **≥ 180 min:** two improvement waves; hard feature freeze at T−40; final submit by T−20; report by the 10% safety margin (T−18 for 180 min).
- **120 min:** use the core cycle in §3.2 exactly; reserve wave only if it can finish by T+100.
- **CTF/flag tasks:** ignore the ML columns; use the fast path below.

### 3.4 CTF / Flag Fast Path

1. **0–2 min:** read task, extract flag format (`LMML{...}`, `flag{...}`, custom), input type, allowed hints.
2. **2–8 min:** cheap hypotheses in parallel: `strings`, file magic, EXIF, archives, stego, entropy,
   pickle/safetensors, Unicode/homoglyph, prompt injection, cipher/encoding, source code comments.
3. **8–12 min:** test hypotheses locally; do not submit guesses.
4. **12–16 min:** submit the best confirmed flag after format check; record response.
5. **16–20 min:** report; if the flag is accepted, collect the clue/passphrase and close.
- Brute force only when the task explicitly permits it and the API is not rate-limited.
- Prompt tasks: each prompt is evaluated independently; make one strong, specific attempt. Do not spam.
  Prompt submissions use the same `POST /submission/text` endpoint (max 15,000 characters); only the latest counts.

**Deadline caveat:** if the task has no `deadlineAt` (Default realm may be untimed), use the global
submissions close from `/settings` and a self-imposed **60-min budget**. The same milestones apply;
the server clock is still authoritative.

### 3.5 Checkpoint Discipline

Every phase gate writes a 3-line status to the orchestrator. A missed checkpoint triggers:
ping at +5 min, reassign at +10 min. The orchestrator does not wait silently for an agent.

### 3.6 Frozen / Antarctica Chain (Conditional)

If the team completes the 10-word passphrase and enters the Frozen realm, treat it as a
**separate 5-stage sequential timed chain**, not as part of the 24-task Default plan:

- One dedicated operator + one solver; no unrelated task work while a stage is active.
- Each stage has its own `deadlineAt`; 10-minute red alert. Baseline by 50% of the stage limit.
- Do not enter until the toolchain for all five domains is ready. Leaving is permanent.
- Frozen points add to the same team score, but a broken/unseeded chain must not cost Default points.
- If a stage is unseeded or broken, report to the organizers and return to Default work.
- Verify the chain mechanics live (stage count, deadline, next unlock) before passing the gateway.

---

## 4. Parallelization and Compute Budget

### 4.1 What Parallelism Buys

| Resource | Parallel agents help? | Rule |
| :--- | :--- | :--- |
| Thinking / hypothesis generation | Yes | Up to 4 cheap agents |
| CPU training / CV | **No** | One heavy job; agents share the same 4 cores |
| I/O / downloads / EDA | Somewhat | One downloader + one reader max |
| Validation / report writing | Yes | Reviewer + scribe can run while solver trains |
| API submission | No | Single submission officer; serialize calls |

### 4.2 CPU / RAM / Disk Budget

Every compute shell must export:

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
       NUMEXPR_NUM_THREADS=2 POLARS_MAX_THREADS=2
```

- LightGBM/XGBoost/CatBoost: `num_threads=2`, `n_jobs=2`; sklearn `n_jobs=2`; torch `set_num_threads(2)`.
- One **heavy** job (2 threads, full CV) + one **light** job (1 thread, EDA/feature work) max.
- Never run two full-data CVs, two Optuna sweeps, or two neural trainings simultaneously.
- RAM: target ≤ 6 GB per job; keep ≥ 2 GB free; kill at 12 GB RSS; use Polars lazy/scan, parquet, `np.memmap`.
- Disk: ≤ 10 GB per task; delete caches at closeout; no > 20 GB download without orchestrator approval.
- Before spawning a solver, run `free -g` and `ps -eo pid,rss,comm --sort=-rss | head`.

### 4.3 Concurrency Matrix

| Situation | Solvers | Reviewer | Intake | Scribe | Heavy jobs |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Intake burst | 0–1 | 0 | 1 | 0 | 0 |
| Baseline | 1 | standby | 0–1 | 0 | 1 |
| Improvement | 1 (+1 light) | 1 | 0 | 0 | 1 |
| Validation | 0 | 1 | 0 | 1 | 0–1 light |
| Long training | 1 | 1 | 0 | 1 | 1 |
| CTF / flag | 2 cheap | 1 | 1 (may double as a tester) | 0 | 0 |
| Crisis (deadline < 15 min) | 1 | 0 | 0 | 0 | 1 |

**Never exceed:** 4 subagents, 2 heavy jobs, 1 Optuna sweep, 1 full-data CV, 1 API submit in flight.

### 4.4 One Strong Pipeline vs Independent Algorithms

**Default: one strong pipeline.** On 4 cores, depth in one pipeline beats breadth across three.
Add an independent solver only when at least two are true:

- Baseline is below 60% of `maxScore`, or clearly below the visible leaderboard median.
- The task is genuinely multimodal (text + tabular, image + metadata, audio + transcript).
- The independent candidate is cheap: < 10 min, < 2 GB RAM, < 1 core sustained.
- The primary solver is blocked on a long job and the second can work on a different feature view.
- The final 30 min need ensemble diversity and OOF predictions already exist.

**Do not:** add a third model family, tune hyperparameters before a valid baseline, run a
full ablation grid, or ensemble models whose OOF does not beat the best single model.
Prefer seed diversity and feature-view diversity over architecture diversity on CPU.

**If several tasks are open at once:** one primary task per solver. A second solver may bank a
cheap baseline on another task only while no heavy job is running. Never optimize two tasks on
one 4-core box.

### 4.5 Spawn / Kill Rules

- Spawn only with: one-page brief, owned path, deadline, output artifact, stop condition.
- One solver per sandbox path. Two agents must never write the same `runs/<run_id>/`.
- Reviewer is assigned at T+10 but starts work only when a candidate exists; idle reviewers waste context.
- Kill after: 10 min with no artifact, 2× ETA, repeated identical error, or scope creep into another track.
- Respawn with a **narrower** scope, not the same prompt. Preserve logs before killing.
- Reuse a session across adjacent phases only when its context is still valid; start fresh when switching tasks.

---

## 5. File Ownership and Commit Protocol

### 5.1 Ownership Table

| Path | Writer | Notes |
| :--- | :--- | :--- |
| `work/tasks/<task_id>/` | Intake, solver, reviewer, submission officer (own subdirs) | Gitignored. All code, data, models, logs, predictions live here. |
| `work/tasks/<task_id>/runs/<run_id>/` | Exactly one solver | No other agent writes here. |
| `work/tasks/<task_id>/validation/` | Reviewer only | Reproduction scripts + logs. |
| `work/tasks/<task_id>/submissions/` | Submission officer only | Candidate copies + `ledger.jsonl`. |
| `reports/levels/<id>_BRIEF.md` | Intake (draft), orchestrator (approve) | The only intake output that enters git. |
| `reports/levels/<id>_VALIDATION.md` | Reviewer | Signed PASS/FAIL. |
| `reports/levels/<id>_SUBMISSIONS.md` | Submission officer | Human-readable ledger summary. |
| `reports/levels/<id>.md` | Scribe | Final `TASK_REPORT`. |
| `reports/levels/INDEX.md` | `tools/intake.py` only | Auto-generated index; no hand edits. |
| `reports/levels/LESSONS.md` | Scribe | Append-only reusable lessons; never rewrite history. |
| `reports/LIVE_STATUS.md` | `tools/watch_state.py` only | Never hand-edit; watcher owns it. |
| `reports/LOG.md` | Orchestrator only | Append-only event log. |
| `reports/BOARD.md` | Orchestrator only | Optional live task board; do not duplicate the same table in `LOG.md`. |
| `lvl_XX/**` | Orchestrator only | Promotion destination; subagents never write here. `RESULT.json` and `submissions/` are gitignored runtime artifacts. |
| `templates/**`, `optimizations/**`, `tools/**` | Environment fixer with orchestrator approval | Any change requires `lmml_master.py test-templates` or the relevant self-test. |

**Subagent rule:** subagents write **only** inside their owned `work/` sandbox and their
assigned report file. The only subagent artifact that enters git is a report. Code is promoted
by the orchestrator. No subagent runs `git add`, `git commit`, `git push`, `git checkout`, or
`git reset` — and no subagent runs `tools/report_commit.py` or `git_sync.py` either.

### 5.2 Sandbox Layout

```
work/tasks/<task_id>/
  task.json                  # raw API response
  summary.md                 # intake normalization
  files/                     # downloaded attachments (gitignored)
  manifest.json              # inventory + hashes + shapes
  brief.md                   # working copy of TASK_BRIEF
  runs/
    r01_baseline/
      solve.py  config.json  run.log  RESULT.json
      submissions/submission.csv
      oof.npy  test.npy
    r02_improved/
  validation/
    VALIDATION.md  validation.json  repro.sh  repro.log
  submissions/
    ledger.jsonl
    best_<score>_<sha8>.<ext>
  closeout.json
```

**Convention conflict:** `reports/03_PLAYBOOK_CPU.md` uses `lvl_XX/data/` for raw data; this
plan uses `work/tasks/<id>/files/` for experiments. Pick one convention per task, pass absolute
paths to templates, and never duplicate large datasets. `lvl_XX/` is for the promoted solver only.

### 5.3 Promotion to `lvl_XX`

Only after reviewer PASS + submission accepted. The orchestrator copies:

- `work/tasks/<id>/runs/<run_id>/solve.py` → `lvl_XX/solve.py` as a **thin wrapper**
  (template + exact CLI args + seed), not a 2,000-line template dump.
- Best candidate → `lvl_XX/submissions/submission.csv` (or the task's required filename).
- Normalized result → `lvl_XX/RESULT.json`.
- Metadata → `lvl_XX/config.json` (track, metric, direction, task_id, run_id, sha256).

Header required in `solve.py`:

```python
# Promoted from work/tasks/<task_id>/runs/<run_id> @ <sha256-8>
# Repro: /home/kozak/mllm/py lvl_XX/solve.py <exact args>
```

**Repo policy:** `.gitignore` ignores `RESULT.json` and `submissions/` at every level. Do not
force-add them. They are runtime artifacts for `lmml_master.py`/CI; the durable record is the
report + ledger + hash. Commit only `lvl_XX/solve.py` and `lvl_XX/config.json` if they are tracked.
For a submission > 20 MB, keep it in `work/tasks/<id>/submissions/` plus an outside backup and
reference its path + SHA256 in the report; never put it in `reports/`.

### 5.4 Commit Protocol (Orchestrator Only)

- Work on `main`. No long-lived branches, no force-push, no history rewrite.
- If isolation is unavoidable, use a local-only `scratch/<task>-<agent>` branch, rebase onto `main`, delete it after the commit; never push it.
- Commit windows: (1) brief approved, (2) baseline accepted, (3) final accepted, (4) hourly docs.
- One committer at a time; wait until active writers finish the file.

**Path A — reports, tools, docs (preferred):** use the repo's safe tool.

```bash
cd /home/kozak/mllm/MML
/home/kozak/mllm/py tools/report_commit.py --dry-run
/home/kozak/mllm/py tools/report_commit.py -m "task(<country>): baseline cv=0.812 | lgbm 5f | sub #1234 passed"
```

`report_commit.py` stages only `reports/`, `tools/`, `.gitignore`; it locks, fetches/rebases,
scans for secrets/token-like filenames, refuses files > 20 MB, and never force-pushes.

**Path B — level code promotion (rare, explicit paths):**

```bash
cd /home/kozak/mllm/MML
git status --short
git add lvl_XX/solve.py lvl_XX/config.json
git diff --cached --stat
git commit -m "task(<country>): promote solve cv=0.847 | lgbm+cat blend"
git pull --rebase origin main
git push origin main
```

Never stage `RESULT.json`, `submissions/`, `work/`, data, weights, `.vault/`, `.env`, or tokens —
they are ignored by policy; do not use `git add -f`. Never `git add -A` or `git commit -a`.

**Path C — scoreboard:** run `git_sync.py` only at a quiet checkpoint, orchestrator-only.
Its `--dry-run` is **not read-only** (it rewrites `README.md` and stages everything); never run it
while other agents are writing. Prefer `report_commit.py` for normal report/tool commits.

Commit message format:
`task(<country>): <phase> cv=<score> | <model/blend> | sub #<id> <status>`
Examples:
- `task(japan): brief + baseline cv=0.812 | lgbm+ridge 5f | sub #1234 passed 612`
- `task(japan): final cv=0.847 | lgbm+cat blend | sub #1240 passed 740`
- `docs: hourly status H+06 | 7/24 tasks closed`

Pre-commit gate:
1. `git status --short` reviewed; only intended files.
2. No `work/`, `data/`, `.vault/`, weights, archives, `.env`, tokens; reports ≤ 20 MB; repo hard stop 50 MB.
3. `report_commit.py` preflight passes (Path A) or explicit-path diff reviewed (Path B).
4. If a submission is included: `lmml_master.py check` PASS.
5. If code is included: `/home/kozak/mllm/py -m compileall` on changed Python, plus relevant self-test.
6. `git pull --rebase origin main` before push; push only after a clean rebase.
7. Optional local tag: `git tag task-<id>-best-<score>` (do not push tags unless useful).

**No subagent pushes. No subagent commits. No exceptions for "small fixes".**

### 5.5 Conflict Handling

- Other participants push to the same repo. Before any commit: `git pull --rebase origin main`.
- If `tools/report_commit.py` exits 3, it aborted a conflicted rebase; resolve the named files or defer. Do not force or reuse a stale index.
- Conflict in `README.md` (auto-generated): do not hand-merge; run `git_sync.py` only at a quiet checkpoint, or leave it to the next scoreboard update.
- Conflict in another participant's file: abort the rebase, keep your work in `work/`, ask the human captain. Never overwrite.
- Keep the working tree clean before switching tasks; a dirty tree plus a new task is how work gets lost.
- If rebase is impossible, commit locally and continue; do not force-push. Resolve with the human at the next checkpoint.

---

## 6. Handoff Artifacts

### 6.1 Artifact Table

| Artifact | Path | Producer | Consumer | Required content |
| :--- | :--- | :--- | :--- | :--- |
| Task brief | `reports/levels/<id>_BRIEF.md` | Intake | Solver, reviewer, officer | Metric, format, fair play, deadline, data, plan |
| Data manifest | `work/tasks/<id>/manifest.json` | Intake | Solver, reviewer | Files, sizes, hashes, rows, cols, dtypes, target |
| Baseline result | `work/tasks/<id>/runs/r01_baseline/RESULT.json` | Solver | Reviewer, officer | `cv_score`, metric, submission path, config, hash |
| Candidate bundle | `work/tasks/<id>/runs/<run_id>/` | Solver | Reviewer, officer | Code, log, config, OOF/test, submission, RESULT |
| Validation log | `reports/levels/<id>_VALIDATION.md` | Reviewer | Officer, orchestrator | PASS/FAIL, checks, reproduction, leakage audit |
| Submission ledger | `work/tasks/<id>/submissions/ledger.jsonl` | Submission officer | Orchestrator, scribe | Every attempt: id, hash, status, score, time |
| Final report | `reports/levels/<id>.md` | Scribe | Orchestrator, future tasks | Goal, data, approach, validation, score, lessons |
| Closeout | `work/tasks/<id>/closeout.json` | Scribe/orchestrator | Orchestrator | Best hash, best score, paths, state, clue collected |

### 6.2 Data Manifest Schema

```json
{
  "task_id": "task_japan",
  "fetched_at": "2026-11-14T10:05:00Z",
  "source_urls": ["..."],
  "files": [
    {"name": "train.csv", "path": "files/train.csv", "bytes": 1234567,
     "sha256": "...", "rows": 10000, "cols": 42,
     "dtypes": {"id": "int64", "target": "int8"}, "notes": ""}
  ],
  "sample_submission": {"path": "files/sample_submission.csv", "rows": 5000, "cols": 2},
  "target_col": "target",
  "id_col": "id",
  "metric": "roc_auc",
  "direction": "max",
  "missing_or_blocked": []
}
```

### 6.3 `RESULT.json` Minimum Schema

Templates already write most of this; the orchestrator normalizes before promotion.

```json
{
  "run_id": "r02_improved",
  "task_id": "task_japan",
  "track": "tabular",
  "task": "binary",
  "metric": "roc_auc",
  "direction": "max",
  "cv_score": 0.847,
  "score": 0.847,
  "cv_scheme": "stratified_kfold(5)",
  "submission": "submissions/submission.csv",
  "sha256": "...",
  "n_train": 10000,
  "n_test": 5000,
  "model": "lgbm+catboost blend",
  "seed": 42,
  "elapsed_s": 1234.5,
  "fair_play": {"pretrained": false, "external_data": false, "external_api": false},
  "notes": "one line"
}
```

`lmml_master.py` reads `cv_score` (preferred) or `score`; keep both when possible.
A result without a submission hash is not accepted for promotion.

### 6.4 Submission Ledger Schema (`ledger.jsonl`)

```json
{"ts":"2026-11-14T10:34:12Z","task_id":"task_japan","candidate":"r01_baseline",
 "file":"submissions/submission.csv","bytes":81234,"sha256":"...",
 "api_submission_id":"1234","status":"passed","score":612,"max_score":1000,
 "feedback":"...","submitted_by":"officer"}
```

Append-only. Never rewrite an entry. If the API returns an error, log it with `status:"error"`.

### 6.5 Handoff Message Format

```
HANDOFF <task_id> | <from> -> <to>
STATE: <phase> | best=<score>/<max> | candidate=<run_id> sha=<sha8>
ARTIFACTS: <paths>
ASSUMPTIONS: <metric, fair play, CV scheme>
NEXT: <one action> | DEADLINE: <UTC> | RISK: <one line>
```

---

## 7. Escalation Rules

### 7.1 Escalation Matrix

| Sev | Trigger | Detect | Owner | First action | Timebox | Fallback |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **P0** | Deadline near / task expired | `nextRemaining` < 10 min; 403 on submit | Orchestrator | Freeze code; submit best-known file; stop new work | 5 min | Close task; report; move on |
| **P0** | API 401 | Any call returns 401; watcher alert | Orchestrator | Stop submissions; refresh JWT; verify `/auth` | 10 min | Continue compute; queue submissions |
| **P1** | Submission pending / throttled | `submissionPending`, `submissionThrottled` | Submission officer | Check `/submission/team/my`; back off 10/30/60 s | 10 min | Wait; write report; retry once |
| **P1** | Task locked / 403 | `task access is denied`, `taskNotInRealm` | Orchestrator | Check journey `nextTaskId`; solve prerequisite | 5 min | Wait; prep from public text |
| **P1** | Missing / corrupt data | 404, empty archive, bad magic, expired URL | Intake + env fixer | Re-run intake `--force`; re-fetch URL; inspect archive | 10 min | Solve from description; submit partial |
| **P1** | Metric / format ambiguity | Sample doesn't match description; score erratic | Intake + reviewer | Probe with baseline; compare CV vs LB; document assumption | 15 min | Optimize safest proxy; submit valid baseline |
| **P1** | Fair-play ambiguity | Task text silent/contradictory | Intake + orchestrator | Default deny; quote task text; choose from-scratch | 10 min | Conservative path only; document |
| **P1** | Score regression / rejection | New score < previous; status rejected | Submission officer | Re-submit last passed file if time | 5 min | Keep backup; report; no further attempts |
| **P2** | Environment / import error | Traceback, `libz.so.1`, missing module | Env fixer | Run `py` wrapper; `LMML_NO_RUST=1`; reinstall minimal | 15 min | Pure-NumPy/CPU fallback |
| **P2** | OOM / CPU thrash | RSS > 12 GB; load > 8; swap | Env fixer | Kill own job; cut folds/batch; free caches | 10 min | Subsample; single model |
| **P2** | Agent stalled | No artifact 10 min; 2× ETA | Orchestrator | Ping; inspect log; reassign narrower | 10 min | Kill; do task directly |
| **P2** | Hint penalty decision | Hint costs points | Orchestrator | Unlock only if expected gain > penalty and ≥ 20 min left | 2 min | Skip hint |
| **P3** | Clue/passphrase slot | After task close | Scribe | Record clue; 2 min max | 2 min | Defer to H18 consolidation |

### 7.2 Detailed Playbooks

**API 401 (token expired)**
1. Stop all submission calls immediately; do not retry in a loop.
2. Orchestrator checks JWT expiry without printing the token:
   `/home/kozak/mllm/py -c "import base64,json,time; t=open('/home/kozak/mllm/.sigmoid_token').read().strip(); p=t.split('.')[1]; p+='='*(-len(p)%4); print(json.loads(base64.urlsafe_b64decode(p))['exp'], time.time())"`
3. Refresh from the portal session (browser localStorage `auth-token`) or ask the human captain.
4. Write the new token to `/home/kozak/mllm/.sigmoid_token`, `chmod 600`; never paste it into a report or shell history.
5. Verify: `tools/sigmoid_api.py me` and `settings`.
6. Keep modeling while auth is being fixed; queue at most one submission candidate.

**403 task access denied**
- Before unlock: expected. Do not poll `/task` faster than every 20 s; the watcher already polls.
- After unlock: check `realm-journey/my` for `nextTaskId`; you may be on the wrong realm or trying a task out of sequence.
- `taskNotUnlocked`: a future sequential stage; solve the previous stage first.
- `taskTimeExpired`: the task deadline passed; stop submissions and hints.
- `clueLocked`: attempt limit exhausted; the clue is already granted; stop submitting.
- `realmLeft`: the team left the realm; it cannot be re-entered without an admin reset.
- `submissionsClosed`: the global window closed; stop all submissions and finalize reports.
- If the task is not in the current realm, switch realm in the UI or solve the prerequisite. Record the blocker; move to the next available task.

**Submission pending / throttled**
- Parse the error body; check `GET /submission/team/my?limit=5` and the pending endpoint.
- A pending evaluation is **team-wide**: no teammate may submit to that task until it clears.
- Wait 10 s, then 30 s, then 60 s. One retry per window. Never double-POST the same file.
- While waiting: write the report, prepare the next candidate, or collect the clue. Do not start a new training job.
- If pending > 10 min, notify the human captain; the evaluator may be stuck. Keep the file backed up.
- `evaluatorUnavailable` / `evaluationFailed`: the submission was not counted; keep the file and retry once when the evaluator recovers. Do not spam.

**Missing data**
- Re-run `tools/intake.py --task-id <id> --force` to refresh presigned URLs.
- Check `task.json` for URLs hidden in nested keys; check `files/` for zero-byte or HTML-error downloads.
- Verify magic bytes, archive integrity (`unzip -t`, `tar -tf`), password hints in the description.
- If the source is Kaggle/Drive/GitHub, use the provided link only; do not substitute external data unless fair play allows it.
- If data is too large for RAM/disk, stream/subsample or use Polars scan; never download > 20 GB without approval.
- After 10 min: submit the best description-based/partial answer, write the blocker, close the task.

**Metric ambiguity**
- Read the sample submission and evaluator feedback; identify whether the score is higher-is-better.
- Submit one cheap baseline to probe the evaluator (the score is feedback, not a final commitment).
- If the metric is still unclear, optimize a monotone proxy (AUC ≈ rank; logloss ≈ calibrated probability; F1 ≈ threshold sweep; RMSE ≈ MAE).
- Document the assumption in the brief and validation log. Do not chase LB noise with repeated submissions.

**Fair-play uncertainty**
- Search the task text for: pretrained, pre-trained, foundation model, external data, external dataset, internet, API, download, Hugging Face, ImageNet.
- If the text is silent: **deny**. Use from-scratch models, classical features, and provided data only.
- If explicitly allowed: enable only the named resource and record the exact quote in the brief.
- Never pass `--pretrained` or `--allow-external-api` "just in case". A disqualification costs all points.
- Never share flags, weights, code, or hints outside the team.

**Time expiry**
- `hard_stop = min(task.deadlineAt or global_close, journey.nextDeadlineAt or global_close) − max(5 min, 10% of remaining)`.
- If the task is untimed (Default realm), use `settings.submissionsWillCloseAt` as `global_close` and a self-imposed 60-min budget.
- At `hard_stop − 10 min`: freeze code and feature changes.
- At `hard_stop − 5 min`: final submission call only; no new uploads.
- If an upload is in progress at `hard_stop − 2 min` and may not finish, abort and submit the smaller fallback.
- At expiry: record `EXPIRED`, keep the last counted submission, write the report. Do not attempt to bypass the timer.

**Hints**
- `GET /hint/task-overview/:taskId` reveals the penalty schedule **without unlocking**; collect it for every task at intake.
- Unlock only if the expected gain exceeds the penalty and ≥ 20 min remain. Penalties are per task and irreversible.
- On clue-linked tasks, the attempt limit may be hidden; never burn rejects for a word unless the task's points are already lost or the word is worth more.

**Score regression**
- The latest submission is the counted one. Before submitting, compare against the current best score/hash.
- Keep the exact bytes of every passed submission: `submissions/best_<score>_<sha8>.<ext>`.
- If a worse candidate is accepted, immediately re-submit the best file (if time and no pending evaluation).
- If a candidate is rejected, the previous passed submission should still count, but verify in `/submission/team/my`; re-submit the best file if the ledger is ambiguous.

**Environment / OOM**
- Reproduce with `/home/kozak/mllm/py`; never use system Python.
- `LMML_NO_RUST=1` for the pure-NumPy fallback; rebuild Rust only via `lmml_master.py rust-build`.
- Check `free -g`, `ps -eo pid,rss,comm --sort=-rss | head`, `df -h .`.
- Kill only your own processes. Reduce folds, batch size, image size, or sample count; free `/tmp` and model caches.
- If the environment is broken > 15 min, switch to a pure-Python baseline and report the incident.

---

## 8. First 15 Minutes After the Journey Unlocks

### 8.1 T−10 Preflight (once, before unlock)

Run these so the first 15 minutes are pure task work:

1. `cd /home/kozak/mllm/MML`
2. `/home/kozak/mllm/py tools/watch_state.py --once` — confirm JWT, settings, journey null/live.
3. `/home/kozak/mllm/py lmml_master.py health --no-network` — Python, Rust, disk, env.
4. `/home/kozak/mllm/py -c "import numpy,polars,sklearn,lightgbm,xgboost,catboost,torch; print('ok')"`
5. `nproc; free -g; df -h .` — record baseline; confirm 4 cores / ≥ 8 GB free.
6. Start the watcher: remove `work/state/stop`, then `nohup /home/kozak/mllm/py tools/watch_state.py --interval 10 > work/state/watch.out 2>&1 &`
7. Confirm `reports/LIVE_STATUS.md` updates and `work/state/events.jsonl` grows.
8. Prepare `reports/levels/` + `work/tasks/` directories; open `TASK_BRIEF.md`, `SUBMISSION_CHECKLIST.md`.
9. Verify a template smoke test (one fast `--demo`) so imports/libs are warm.
10. Sync clocks: `date -u`; compare with `serverNow` from the journey payload; note offset.
11. Write the preflight line in `reports/LOG.md` (orchestrator).
12. Assign roles for the first task: intake, solver, reviewer, submission officer; scribe on standby (spawn at T+60 or after final submit).

### 8.2 T+0 → T+15 Step-by-Step

| Min | Action | Owner | Output / gate |
| :--- | :--- | :--- | :--- |
| **T+0** | Watcher fires `[unlock]`. Run `tools/intake.py` immediately (skip re-download if the watcher already refreshed `work/tasks/index.json` in the last 60 s). | Orchestrator | `work/tasks/index.json`, `reports/levels/INDEX.md` |
| **T+0–1** | `tools/sigmoid_api.py journey` and `settings`; capture `nextTaskId`, `nextDeadlineAt`, `nextRemaining`. | Orchestrator | Deadline in UTC written to `LOG.md` and brief |
| **T+1–3** | Open `task.json`; extract title, description, input type, max score, complexity, max upload size, accepted extensions, hints, metric, sample, `realm`, `timeLimitMinutes`, `countryCode`; make the fair-play decision. | Intake | `TASK_BRIEF.md` v0 (one page) |
| **T+3–5** | Download/normalize attachments; build `manifest.json` (size, sha256, rows, cols, dtypes, target, sample). | Intake | Manifest + missing-data flags |
| **T+5–8** | Run EDA + CTF forensics: `fast_eda.py`, `ctf_tactics.py`, `lmml_rs eda` on the main CSV. | Intake | 5-line findings; baseline plan |
| **T+8** | Spawn solver with brief + baseline command + stop condition. | Orchestrator | Solver session live |
| **T+8–10** | Spawn/assign reviewer (standby) and notify submission officer. | Orchestrator | Roles confirmed |
| **T+10–15** | Baseline running; prepare fallback file; validate sample format; start report stub. | Solver + officer | Baseline ETA; G0/G1 passed |

**If intake returns 403 at T+0:** wait 2 s and retry once; the task endpoint may lag the
`allowSubmissions` toggle. Do not loop or assume the task is missing.

**If the unlock is a later task (N > 1):** skip the long preflight. Do a 2-minute check
(`free -g`, watcher alive, `py` works), then run the same T+0–T+15 sequence. Reuse the
role sessions from the previous task only if their context is still relevant; otherwise
spawn fresh with the new brief.

**If the journey does not unlock when expected:** do not poll `/task` in a tight loop.
Check `settings` (`allowSubmissions`, `submissionsWillOpenAt`), `journey`, and the watcher log.
Prepare recon from public descriptions, run template demos, and record the delay. The server
clock is the only source of truth; local time is not.

### 8.3 First-Task Extra Checks

- Confirm the actual task count (`GET /task?limit=100` → `totalItems`) and country codes. If it is
  materially more than 24, switch to triage: easy/medium + high `maxScore` first; baseline only
  tasks you can reach; record skipped tasks with a reason.
- Confirm the task-slot mapping: `task_id` → `lvl_XX` (unlock order). Write it in the brief and the task board.
- Confirm submission input type by reading the task UI/API, not by guessing from the template.
- Confirm whether the task expects a flag, a file, or a prompt; a CSV template is useless for a flag task.
- Confirm fair play in the first 5 minutes; this decision cannot be reversed cheaply later.
- Confirm the evaluator loop: submit the baseline early enough to learn the feedback format before the improvement wave.

---

## 9. Reporting Cadence and Status Format

### 9.1 Cadence

| Cadence | When | Owner | Content |
| :--- | :--- | :--- | :--- |
| **Event** | Task open, baseline, candidate, submit, close | Respective owner | One status line each; append to `LOG.md` for milestones |
| **10-min** | During active task | Solver | DONE / DOING / BLOCKED / ETA / FILES |
| **Hourly** | Every 60 min from journey start (e.g., :55 UTC if start is 08:00 UTC) | Orchestrator | Task board snapshot + burn rate + risks |
| **Daily** | H0, H12, H24 of the journey | Orchestrator | Tasks passed, points/max, best scores, top 3 risks, next 6 h plan |
| **On-call** | Any P0/P1 | Owner | Immediate escalation in the status format |

The "daily" rhythm in a 24 h event means three briefs: start, midpoint, final. Do not write essays.

Hourly heartbeat example:

```
H+06 | closed 6/24 | pts 2340/24000 | active task_chile (dl 12:40, best 0) | P0 none | next: baseline task_belgium
```

### 9.2 Compact Status Line

```
[HH:MMZ] T<NN> <task_id> | <PHASE> | best=<score>/<max> | sub=#<id>:<status> | cv=<x> | cpu=<n>/4 | mem=<n>G | eta=<m> | blocker=<none|...> | next=<...>
```

Example:

```
[14:05Z] T07 task_japan | IMPROVING | best=612/1000 | sub=#1234:PASSED | cv=0.842 | cpu=2/4 | mem=6.1G | eta=12m | blocker=none | next=validate blend
```

Rules: ≤ 200 characters; no secrets; no raw logs; always include the deadline if < 20 min remain.

### 9.3 Agent Reply Format

```
STATUS <role> <task_id> | DONE: <artifact + result> | DOING: <next> | BLOCKED: <none|reason> | ETA: <min> | FILES: <paths>
```

Example:

```
STATUS solver task_japan | DONE: r02 cv=0.847 | DOING: audit blend | BLOCKED: none | ETA: 12m | FILES: work/tasks/task_japan/runs/r02
```

- `DONE` must name a file and a number (score, rows, exit code), not "finished".
- `BLOCKED` must name the exact error and what was tried.
- `ETA` is minutes from now; if it changes by > 50%, send an update.
- No multi-paragraph reports in chat; put details in the assigned report file.

### 9.4 Task Board

Maintain one Markdown table in `reports/LOG.md` (or a dedicated `reports/BOARD.md` if the
orchestrator prefers; do not create both):

| Slot | Task | State | Best / Max | Last sub | CV | Deadline (UTC) | Owner | Blocker |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 01 | `task_...` | CLOSED | 740/1000 | #1240 PASSED | 0.847 | 10:55 | — | — |
| 02 | `task_...` | IMPROVING | 0/1000 | — | — | 12:40 | solver B | none |

Update on every state transition. The board is the orchestrator's source of truth when
deciding whether to spend the reserve wave or move on.

### 9.5 What Not to Report

- Tokens, cookies, API keys, private URLs with signatures.
- Raw task data or full submission contents.
- Unverified claims ("should work", "probably passes").
- Enormous logs; link the path and quote the last 10 lines.
- Other teams' information or anything obtained outside the team's account.

---

## 10. 24-Hour Rhythm and Prioritization

| Window | Goal | Rules |
| :--- | :--- | :--- |
| **H0–H2** | Unlock, first 1–2 tasks, validate the whole pipeline (intake → baseline → submit → report) | Do not optimize; prove the loop works. Record real task durations vs plan. |
| **H2–H6** | Breadth: every unlocked task gets a valid baseline | Bank first. Skip depth if it risks the next task's baseline. |
| **H6–H12** | Depth on high `maxScore` / high-gap tasks | One reserve wave per task maximum. Re-check fair play before any pretrained/API use. |
| **H12–H18** | Consolidation: fix rejected/pending submissions, revisit weak tasks, collect clues | Do not start new model families after H18 unless a task is clearly high-value. |
| **H18–H22** | Reliability: ensure every still-open task's latest submission is the best-known; finish reports | Expired tasks cannot be resubmitted; verify their final counted score instead. |
| **H22–H24** | Freeze and final audit | No new modeling after H23. Verify all ledger entries, report links, git state. |

**Prioritization rule:** expected points per minute = `maxScore × P(improvement) / minutes`.
Because tasks are sequential, you cannot reorder them; you can only decide how much of the
reserve wave to spend. Default: baseline on every task, reserve wave only when the expected
gain is > 10% of `maxScore` and the risk to the counted submission is zero.

**Clue/passphrase:** collect one clue per closed task if offered; cap at 2 min per task.
The 10 clue slots are Argentina, Australia, Belgium, Chile, France, Japan, New Zealand,
Norway, South African Republic, and United Kingdom. A pass (or enough rejects) unlocks the word;
never burn attempts unless the word is worth more than the task's points. The Antarctica
meta-puzzle is a bonus, not a reason to miss a baseline.

---

## 11. Do-Not-Do List

1. Do not run `git add -A`, `git commit -a`, `git push`, `git reset --hard`, `git clean -fd`, `tools/report_commit.py`, or `git_sync.py` as a subagent.
2. Do not write to `lvl_XX/`, `templates/`, `optimizations/`, or `tools/` without orchestrator approval.
3. Do not submit without reviewer PASS and a checklist.
4. Do not submit a worse candidate while the current best is counted; latest wins.
5. Do not enable `--pretrained` / `--allow-external-api` without an explicit task quote.
6. Do not run two heavy jobs, `n_jobs=-1`, or 16 RL envs on 4 cores.
7. Do not download > 20 GB, commit reports > 20 MB or repo files > 50 MB, or put data/weights/tokens in git.
8. Do not poll the API in a tight loop; respect throttling and pending states.
9. Do not spend the reserve wave on hyperparameter tuning with no hypothesis.
10. Do not leave a task without a report, ledger entry, backup file, and closeout.
11. Do not trust `AGENTS.md`'s Windows paths; use the live Linux environment.
12. Do not share flags, weights, code, hints, or credentials outside the team.

---

## 12. Appendix — Spawn Prompt Skeletons

### 12.1 Intake Agent

```
ROLE: intake/recon for <task_id> (slot lvl_XX).
READ: reports/AGENT_CONTEXT.md, reports/templates/TASK_BRIEF.md.
DO: fetch task + attachments (tools/intake.py --task-id <id>), parse inputType/maxScore/
    time limit/max upload/extensions/metric/sample, run fast_eda + ctf_tactics,
    write reports/levels/<id>_BRIEF.md and work/tasks/<id>/manifest.json.
CONSTRAINTS: no modeling, no submissions, no git, default-deny fair play.
DEADLINE: brief ready in 10 min. STOP after handoff message to orchestrator.
```

### 12.2 Track Solver

```
ROLE: <track> solver for <task_id>.
READ: reports/levels/<id>_BRIEF.md and work/tasks/<id>/manifest.json only.
DO: baseline first (template + exact command), then one improvement; save every run under
    work/tasks/<id>/runs/<run_id>/ with run.log, config.json, RESULT.json, OOF/test.
CONSTRAINTS: 2 threads, <=6 GB RAM, no submissions, no lvl_XX edits, no git.
DEADLINE: baseline candidate by T+35; improved candidate by T+55.
REPORT: status line every 10 min; stop at T+70 unless orchestrator extends.
```

### 12.3 Reviewer

```
ROLE: independent validator for <task_id>.
READ: brief, candidate RESULT.json, candidate submission, solver command.
DO: reproduce from a clean shell; audit split/leakage/metric/IDs/NaN/Inf/constant/range;
    run lmml_master.py check or equivalent; write reports/levels/<id>_VALIDATION.md.
CONSTRAINTS: do not modify the candidate; do not submit; do not trust solver claims.
OUTPUT: PASS/FAIL + evidence + exact commands. Deadline: 15 min from candidate freeze.
```

### 12.4 Submission Officer

```
ROLE: submission officer.
READ: reports/templates/SUBMISSION_CHECKLIST.md, brief, validation PASS, ledger.
DO: run every checklist item; verify no pending submission; submit via tools/sigmoid_api.py;
    poll status; append ledger.jsonl; write reports/levels/<id>_SUBMISSIONS.md.
CONSTRAINTS: no submission without reviewer PASS + orchestrator GO; one in flight; back off.
OUTPUT: API id, status, score, hash, or a precise rejection reason.
```

### 12.5 Report Scribe

```
ROLE: report scribe for <task_id>.
READ: brief, manifest, RESULT.json, validation log, submission ledger.
DO: fill reports/templates/TASK_REPORT.md -> reports/levels/<id>.md; record score, submission id,
    CV, fair-play decision, commands, lessons; update closeout.json.
CONSTRAINTS: artifacts only, no memory-based claims; no code changes; no git.
DEADLINE: 10 min after final submit.
```

### 12.6 Environment Fixer

```
ROLE: environment fixer (on-call).
TRIGGER: import/lib/Rust/Playwright/Modal/OOM/disk failure.
DO: reproduce with /home/kozak/mllm/py; smallest fix; verify with the failing command;
    append a 5-line incident note to reports/02_ENVIRONMENT.md and work/env/<ts>.md.
CONSTRAINTS: 15 min timebox; no task-code refactor; no template edits without approval.
OUTPUT: fixed / fallback recommendation / precise blocker.
```

---

## 13. Definition of Done

**Per task — all six must be true:**
1. A valid submission was accepted (or a best-effort submission + documented blocker).
2. `ledger.jsonl` has the API id, hash, status, and score (or `pending` with a follow-up owner).
3. `reports/levels/<id>_VALIDATION.md` is PASS (or FAIL with reason and fallback noted).
4. `reports/levels/<id>.md` is complete and links the artifacts.
5. The best file is backed up under `work/tasks/<id>/submissions/best_<score>_<sha8>.<ext>`.
6. Orchestrator committed the brief/report and promoted solve (if tracked); `RESULT.json`/submission are recorded by path + hash, not force-added.

**Per event:**
- Every assigned slot (`lvl_01..lvl_24`, or the actual `totalItems`) has a state in the board: CLOSED / EXPIRED / SKIPPED with reason.
- Every closed task has a final report and a counted submission.
- `reports/LOG.md` has hourly heartbeats and the H0/H12/H24 briefs.
- No secrets, data, weights; reports ≤ 20 MB, repo hard stop 50 MB.
- Final audit: latest submission = best-known for every task; all reports committed.
