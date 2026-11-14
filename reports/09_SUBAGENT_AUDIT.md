# 09 — Subagent Audit (freeze + git sync)

_Audit by orchestrator, 2026-11-14 ~09:25 UTC, after upstream `origin/main` moved from `3bc4a4a` to `8428d3e`._

## Why this audit

The user asked to stop active subagents, analyze their output, and check what
needs updating locally after the repo changed upstream. Two agents were still
running (environment auditor B, baseline engineer I); both were told to stop
and freeze the tree. The rest had already completed.

## Status table

| Agent | Scope | Deliverables on disk | Verdict |
| :--- | :--- | :--- | :--- |
| **A** | API & intake | `tools/sigmoid_api.py`, `tools/intake.py`, `tools/watch_state.py`, `reports/01_SITE_API_RECON.md` | **Keep.** Working, tested against locked API, no token leaks. Intake schema is inferred until tasks unlock. |
| **B** | Linux/Rust audit | `reports/02_ENVIRONMENT.md`, `tools/env_check.sh` | **Keep.** 6/7 demos pass; security demo failure is a deterministic seed-42 STRIP assertion, passes with `--seed 1`. Report uses the real box (4 threads, 15 GiB), unlike upstream's 20-thread report. |
| **C** | CPU playbooks | `reports/03_PLAYBOOK_CPU.md`, `reports/runbooks/01..07` | **Keep, but revalidate flags.** Written before upstream `4e0946f` rewrote templates (+7.6k lines). High-level strategy is valid; flag names/snippets need a consistency pass. |
| **D** | War-room protocol | `reports/04_WAR_ROOM_PLAN.md`, `reports/templates/*` | **Keep.** Concrete time boxes, ownership, escalation. Aligns with H's latest-submission finding. |
| **F** | Toolkit validation | `reports/05_TOOLKIT_VALIDATION.md` | **Keep with addendum.** Ran before upstream sync; orchestrator re-ran Rust/EDA/CTF/blender/math/cpp/sys_bench after sync — all pass. |
| **G** | Git automation | `tools/report_commit.py`, `reports/06_GIT_WORKFLOW.md` | **Keep.** Dry-run clean, secret scan clean, refuses out-of-scope files, no force push. It found that the *old* `git_sync.py --dry-run` mutated README/index; upstream `4e0946f` fixed the dry-run path (verified by agent L). |
| **H** | Rules/scoring recon | `reports/07_RULES_AND_SCORING.md` | **Keep — highest strategic value.** Critical: latest submission per task counts (not best); hints cost points; default realm has no per-task timer; Frozen realm is a 5-stage timed chain; don't assume 24 tasks (54 country codes). |
| **I** | Auto-baseline | `tools/make_baseline.py`, `tools/validate_submission.py` | **Keep.** 19/19 synthetic layouts pass validation; missing report was written as `08_BASELINE_GENERATOR.md`. |

## Cross-agent issues found

1. **Duplicate API clients.** Upstream `optimizations/sigmoid_api.py` (from
   commit `a447874`) used the wrong upload key (`fileKey` instead of
   `fileStorageKey`), omitted `fileSize`/`fileExtension`, and pointed at a
   non-existent Desktop secrets file. It is used by `orchestrator.py`.
   **Fixed locally:** env/token-file resolution + correct upload contract.
   `tools/sigmoid_api.py` remains the canonical CLI.
2. **Stale environment report.** B's report says `Modal: missing` — Modal is now
   installed (1.6.4), but the token in `/home/kozak/mllm/modal_key` is only the
   `ak-…` token ID; the `as-…` secret is still missing, so cloud GPU is not
   usable yet.
3. **Template drift.** Upstream rewrote all seven templates and Rust core after
   the runbooks were written. The demos and flags must be rechecked; do not
   blindly copy old command lines.
4. **Unsafe `git_sync.py`.** It stages `git add -A` and pushes without a
   fetch/rebase; do not run it concurrently with other participants. Use
   `tools/report_commit.py` for reports/tools.
5. **Security demo flake.** `05_adversarial_sec.py --demo` fails its STRIP AUC
   assertion at seed 42 deterministically; `--seed 1` passes. Not a Linux bug.
6. **Report numbering collision.** Upstream added `reports/01_infrastructure_readiness.md`,
   `02_cloud_and_auth_setup.md`, `03_full_library_verification.md`,
   `04_git_sync_and_deps_update.md`; our agent reports use overlapping numeric
   prefixes but distinct filenames. `reports/README.md` is the index.

## What was updated locally

- Pulled/rebase onto `origin/main` `8428d3e` (3 new upstream commits).
- Rebuilt Rust core against the new `rust_core` (7/7 tests pass).
- Re-ran `rust_fast_ext`, `fast_eda`, `ctf_tactics`, `blender`, `math_tricks`,
  `cpp_fast_ext`, `sys_bench` — all pass.
- Installed the missing `requirements.txt` packages (torch-geometric,
  pennylane, sb3-contrib, scikit-image, faster-whisper, ART, langchain,
  langgraph, kaggle, onnxruntime, etc.); `health` now reports 10/10 packages.
- Patched `optimizations/sigmoid_api.py` upload/token handling.
- Full template demo re-run after sync is in progress; see `10_UPSTREAM_SYNC.md`.

## Recommendation

Use A/D/F/G/H/I outputs as-is; use C's runbooks after the flag revalidation.
Keep `reports/07_RULES_AND_SCORING.md` open during the competition: the
latest-submission rule changes when we should press submit.
