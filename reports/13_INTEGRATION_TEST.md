# 13 — Post-Sync End-to-End Integration Test

_Subagent L, 2026-11-14 ~11:25–11:35 EET. Repo `HEAD=1c59cde` (upstream `4e0946f` + local agent commits), Python `/home/kozak/mllm/py` (3.13.17, venv), 4 vCPU, no GPU._

Scope: verify `lmml_master.py`, `tools/intake.py`, `tools/report_commit.py` and
`git_sync.py --dry-run` after commit `4e0946f` and the dependency update.
Everything ran in throwaway paths (`work/integration_lvl/`, `/tmp/opencode/…`);
no real `lvl_XX` data, token, commit or push was touched.

## Verdict summary

| Workflow | Verdict | Evidence |
| :--- | :---: | :--- |
| `lmml_master.py new --track tabular` | **GREEN** | scaffolded `work/integration_lvl` in 0.26 s |
| `lmml_master.py solve` (CPU, LGB) | **GREEN** | AUC 0.630657, 2.5–4.1 s, `RESULT.json` + submission written |
| `lmml_master.py check --sample` | **GREEN** | 200 rows, 0 nulls/non-finite, ID order matched |
| `lmml_master.py vault` | **GREEN*** | targeted path shows best score; bare `vault` skips `work/` (see F2) |
| `lmml_master.py health` | **GREEN** | 10/10 packages, Rust binary present, CUDA off as expected |
| `tools/intake.py --dry-run` locked API | **GREEN** | clean HTTP 403, exit 0, no writes |
| `tools/intake.py` normalizer/downloader (mock) | **GREEN** | abs+relative downloaded, 404 and oversized handled |
| `tools/report_commit.py --dry-run` | **GREEN** | only `reports/LIVE_STATUS.md` in scope; index/status unchanged |
| `git_sync.py --dry-run` (clone) | **GREEN** | README/level hashes unchanged; non-dry mode still risky (F3) |

`*` = works, with a minor discovery limitation documented below. No red workflows.

## 1. Synthetic tabular challenge

Created `work/integration_lvl/data/` (gitignored) with a seeded
(`numpy.default_rng(20261114)`) binary task:

- `train.csv`: 400 rows × 10 cols (`id`, 6 numeric, 2 categorical, `target`; 55 % positive).
- `test.csv`: 200 rows × 9 cols (no target); `sample_submission.csv`: 200 rows (`id,target`).
- Injected missingness: `age` 20, `income` 32, `score_b` 16, `city` 12 nulls.
- Categoricals: `city` (6 values), `segment` (basic/plus/pro); target is a noisy
  logistic function of score/tenure/segment.

```bash
cd /home/kozak/mllm/MML
/home/kozak/mllm/py lmml_master.py new work/integration_lvl --track tabular --force
```

```
[OK] Scaffolded Europe (tabular) challenge at: /home/kozak/mllm/MML/work/integration_lvl
EXIT=0 RUNTIME_MS=259
```

Created `solve.py` (copy of `templates/02_tabular_fast.py`), `config.json`
(`metric=auc`, `direction=max`), `data/`, `submissions/`, `.vault/`.
Note: `new` resolves an existing path as-is; because `work/integration_lvl`
already existed it stayed under `work/`. If the directory did not exist it
would have been created under `challenges/work/integration_lvl` (name caveat).

## 2. Solve, check, vault, health

```bash
/home/kozak/mllm/py lmml_master.py solve work/integration_lvl -- \
  --train data/train.csv --test data/test.csv \
  --sample-sub data/sample_submission.csv \
  --folds 2 --models lgb --max-trees 60 \
  --no-tune-threshold --no-use-gpu --seed 42
```

```
[TABULAR] auc=0.63066 (single) | models={'lgb': 0.630657} | adv_auc=0.9992 | rust=True | time=0.383s
[RUN:local] /home/kozak/mllm/venv/bin/python …/work/integration_lvl/solve.py --train … --seed 42
[SCORE] 0.630657 (recorded) | time=2.461s
[SUBMIT] file:////home/kozak/mllm/MML/work/integration_lvl/submissions/submission.csv
EXIT=0 RUNTIME_MS=2554
```

First (cold) run: `EXIT=0`, `RUNTIME_MS=4122`, score `NEW BEST!`.
`RESULT.json` is valid: 400 train / 200 test / 71 features / 2 folds,
`oof_artifacts` present, submission header `id,target`, 200 data rows.

The `adv_auc=0.9992 (severe)` line is the template's train-vs-test drift
diagnostic on tiny synthetic data (target-encoding features dominate); it is a
warning, not a gate, and does not affect the workflow.

```bash
/home/kozak/mllm/py lmml_master.py check work/integration_lvl \
  --sample work/integration_lvl/data/sample_submission.csv
```

```
[RUST:lmml_rs] [CHECK] submission.csv: rows=200, cols=["id", "target"], nulls=0, non_finite=0 (0.22 ms)
[PASS] Submission gate passed.
[CHECK] submission.csv: rows=200, cols=['id', 'target'], nulls=0, nans=0, infs=0
[PASS] Submission gate passed.
EXIT=0 RUNTIME_MS=387
```

```bash
/home/kozak/mllm/py lmml_master.py vault                        # EXIT=0, 140 ms
/home/kozak/mllm/py lmml_master.py vault work/integration_lvl    # EXIT=0, 221 ms
```

Bare `vault` prints `[INFO] No vault entries recorded yet.` because it only
scans `lvl_*` and `challenges/`; the targeted form prints:

```
- integration_lvl        best=0.630657 (auc/max) runs=2 | [SUBMIT] …/.vault/best_submission.csv
```

`.vault/vault.json` history and `best_submission.csv` were written correctly.

```bash
/home/kozak/mllm/py lmml_master.py health                       # EXIT=0, 3568 ms
```

`Python 3.13.17`, 4 logical cores, 77.44 GB free, CUDA unavailable (expected),
`lmml_rs` + Rust extension present, **LMML packages 10/10**, all cache hygiene
checks clean. `rustc`/`cargo` show `missing` outside `nix-shell`, but the
prebuilt `rust_core/target/release/lmml_rs` is present and was used
(`rust=True`, `[RUST:lmml_rs]`). Not a regression.

## 3. `tools/intake.py` against the locked API

```bash
/home/kozak/mllm/py tools/intake.py --dry-run
```

```
[intake] GET https://api-prod.sigmoid.academy/api/v1/task (limit=100)
[intake] LOCKED: HTTP 403 {"message":"task access is denied","error":"Forbidden","statusCode":403}
EXIT=0 RUNTIME_MS=348
```

`--json` also returns a clean structured result (`state=locked`, `httpStatus=403`,
`taskCount=0`, exit 0, no paths written). Expected pre-start state; no crash.

### Mock normalizer/downloader

Served `/tmp/opencode/L_intake_mock/www` with `python -m http.server 8765` and
drove `intake.collect_urls` + `intake.run_intake` in-process with the mock task
JSON (SITE_BASE pointed at the local server so relative links resolve locally).
Task contained an absolute URL, a root-relative URL, a 404 URL and a 2 MiB file;
per-file cap was 1 MiB.

| Input | Classification | Download result |
| :--- | :--- | :--- |
| `http://127.0.0.1:8765/abs.csv` | downloadable, `.csv` | **downloaded**, 18 B |
| `/rel.csv` | resolved to `http://127.0.0.1:8765/rel.csv` (+ API fallbacks) | **downloaded**, 20 B |
| `http://127.0.0.1:8765/nope.csv` | downloadable, `.csv` | **failed**, `httpStatus=404`, body snippet kept |
| `http://127.0.0.1:8765/big.bin` | downloadable, `.bin` | **failed**, `too large (2097169 bytes > 1048576)`; rejected from `Content-Length` before write |

Additional checks all passed: no `big.bin` on disk, no `*.part` leftovers,
`task.json`/`summary.md`/`urls.json`/`index.json`/INDEX.md written to the
redirected temp paths, and the run summary reported `downloaded=2, failed=2`.
A second server that streams 2 MiB **without** `Content-Length` was also
rejected by the in-loop cap (`exceeds 1048576 bytes`) with no partial file.

Full CLI test against the same mock (`SIGMOID_API_BASE=http://127.0.0.1:8765/api/v1/`,
static `api/v1/task` page) printed the three URLs in `--dry-run`, including
`https://sigmoid.academy/rel.csv` for the relative link; no task dirs or index
files were created in the repo.

Normalizer unit checks: `files/x.csv` → `https://sigmoid.academy/files/x.csv`
then API fallback; `/files/x.csv` → site, API and API-root candidates;
`mailto:` → no candidates; Markdown links picked up, `github.com` recorded but
not downloadable. Cosmetic: the 404 error string embeds the first 120 chars of
the HTML body (newlines included) in `task.json`; harmless but noisy.

## 4. `tools/report_commit.py --dry-run`

```bash
/home/kozak/mllm/MML $ /home/kozak/mllm/py tools/report_commit.py --dry-run
```

```
[report_commit] safe paths: reports/, tools/, .gitignore
[report_commit] never staged: work/, .vault/, submissions/, data/, venv/, RESULT.json, tokens, weights
[report_commit] mode: DRY RUN (no changes)
[report_commit] git fetch origin main
[report_commit] behind origin/main by 0 commit(s), ahead by 0
[report_commit] DRY RUN: no pull needed (up to date)
[report_commit] changed paths: 1 total, 1 in scope, 0 outside scope
[report_commit] ignored protected entries: .vault/, RESULT.json, __pycache__/, submissions/, work/
[report_commit] would stage 1 file(s):
  + reports/LIVE_STATUS.md
[report_commit] DRY RUN OK - nothing staged, committed, or pushed.
EXIT=0 RUNTIME_MS=808
```

`git status --porcelain` and `git diff --cached --name-only` were byte-identical
before/after. The run only fetches; it does not pull in preview mode.

Path-policy checks (`is_allowed` + `preflight`):

- allowed: `reports/13_INTEGRATION_TEST.md`, `reports/levels/INDEX.md`,
  `tools/intake.py`, `.gitignore`.
- rejected: `work/**`, `venv/**`, `data/**`, `lvl_01/data/**`,
  `submissions/**`, `RESULT.json`, `.vault/**`, `.sigmoid_token`, `gittoken`,
  `*.pt`/`*.safetensors`, and any path containing a `work/` component.
- `preflight("reports/model.safetensors")` → `forbidden file type`; size limit is
  20 MB; secret patterns are scanned. Two layers: `is_allowed` enforces the
  directory allowlist, `preflight` enforces filename/suffix/size/secret rules.

No data, weights, tokens or `work/` content were staged.

## 5. `git_sync.py --dry-run` (in a clone, never the real repo)

```bash
git clone --quiet /home/kozak/mllm/MML /tmp/opencode/L_git_sync   # 5.3 MB
cd /tmp/opencode/L_git_sync
/home/kozak/mllm/py git_sync.py --dry-run
```

```
[git_sync] [dry-run] Would regenerate README.md 24-level scoreboard (disk untouched).
[git_sync] [dry-run] Would stage/commit 1 file(s) (git index & disk untouched):
  + README.md (would regenerate)
EXIT=0 RUNTIME_MS=77
```

`sha256sum -c` confirmed `README.md` and `lvl_01/README.md` unchanged;
`git status` stayed clean. With a level update:

```bash
/home/kozak/mllm/py git_sync.py --dry-run --lvl lvl_01 --score 0.91 \
  --track tabular --status SOLVED --note "integration dry-run"
```

```
Would update lvl_01: status=SOLVED, track=tabular, score=0.91
Would stage/commit 3 file(s): lvl_01/STATUS.json, lvl_01/README.md, README.md
EXIT=0 RUNTIME_MS=81
```

Hashes still matched and `lvl_01/STATUS.json` was **not** created. So the
`--dry-run` code path is non-mutating; the caution in `reports/10_UPSTREAM_SYNC.md`
(“dry-run mutates README/index”) is **not true for the current `git_sync.py`**
(it was likely based on an older revision).

Risk remains only for the **non-dry** path: it calls `git add -A`, then commits
and pushes. Demonstrated safely in the clone by adding one non-ignored file plus
ignored artifacts:

```
[git_sync] [dry-run] Would stage/commit 2 file(s):
  + notes_l_scratch.txt
  + README.md (would regenerate)
$ git add -A --dry-run
add 'notes_l_scratch.txt'
```

`work/integration_lvl/hidden.txt`, `RESULT.json` and `submissions/submission.csv`
were **not** listed because they are gitignored. Therefore: non-dry `git_sync.py`
will commit any *non-ignored* stray file anywhere in the repo (only `.env` names
and files > 50 MB are rejected), and it regenerates root/level READMEs. Keep
using `tools/report_commit.py` for report/tool commits, or run `git_sync.py`
only on a clean, reviewed tree.

## 6. Hygiene and cleanup

- Real repo `README.md` hash equals `git show HEAD:README.md` after all tests.
- No commit, push, token read/print, or write to real `lvl_XX`/`data`.
- Removed pre-existing root demo artifacts (they predated this test; timestamps
  11:09–11:23, sec/tabular `test-templates` output): `RESULT.json`,
  `submissions/` (`adv_samples.npy`, `submission.csv`) and `.vault/`
  (`demo_binary`, `demo_multiclass`, `demo_regression`, OOF/test `.npy`).
- `work/integration_lvl/` retained as disposable evidence; it is gitignored.
- Concurrent agents were active: `reports/LIVE_STATUS.md` was already modified
  and `tools/submit_officer.py` appeared untracked during the session. Neither
  was touched.
- Mock servers (ports 8765/8766) were stopped; all mock data lived in
  `/tmp/opencode/`.

## 7. Findings / follow-ups

- **F1 (info):** `lmml_master.py new` only scaffolds under `work/` when that
  path already exists; otherwise the name is interpreted under `challenges/`.
  Create the data directory first (as done here) or use a `lvl_XX` name.
- **F2 (minor):** bare `lmml_master.py vault` ignores `work/` and
  `challenges/` subdirectories named like `work/integration_lvl`; pass the
  challenge path explicitly. No data loss, vault file is correct.
- **F3 (operational):** `git_sync.py --dry-run` is safe now, but non-dry
  `git add -A` is still repo-wide. Prefer `tools/report_commit.py`.
- **F4 (cosmetic):** intake 404 errors store a truncated HTML snippet in the
  manifest; the HTTP status is captured correctly.
- **F5 (expected):** `health` reports `rustc`/`cargo` missing outside
  `nix-shell`, yet the prebuilt Rust CLI/extension works. `bitsandbytes` remains
  intentionally absent (no CUDA).
