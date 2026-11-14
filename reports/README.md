# LMML 2026 — Work Reports

This directory is the human-readable log of the agent team. Raw task data and
secrets stay outside git (`work/`, `/home/kozak/mllm/.sigmoid_token`).

## Agent reports

| Report | Contents |
| :--- | :--- |
| `01_SITE_API_RECON.md` | sigmoid.academy API, auth, task/journey/submission schema, current state |
| `02_ENVIRONMENT.md` | Linux/NixOS setup, Rust, template demos, wrapper workaround |
| `03_PLAYBOOK_CPU.md` | CPU-only strategy and time-boxed workflow |
| `04_WAR_ROOM_PLAN.md` | Subagent roles, handoffs, task lifecycle, checklists |
| `05_TOOLKIT_VALIDATION.md` | EDA/CTF/blender/math/Rust self-test results |
| `06_GIT_WORKFLOW.md` | Safe multi-agent commit/push protocol |
| `07_RULES_AND_SCORING.md` | Scoring, latest-submission rule, hints, realms, fair play |
| `08_BASELINE_GENERATOR.md` | Offline auto-baseline and submission validator |
| `09_SUBAGENT_AUDIT.md` | Freeze audit of all subagents and conflicts |
| `10_UPSTREAM_SYNC.md` | Git sync, dependency install, post-sync revalidation |
| `11_RUNBOOK_CONSISTENCY.md` | Runbook command/flag audit against the rewritten templates |
| `12_SUBMISSION_OFFICER.md` | Claim + preflight + backup submission workflow |
| `13_INTEGRATION_TEST.md` | End-to-end post-sync `new`/`solve`/`check`/intake tests |
| `14_TASK_ROUTER.md` | Task modality detection, routing, and live board |
| `15_KICKOFF_CHECKLIST.md` | T-15 / T+0 / T+5 / T+10 actions when submissions open |
| `16_MODAL_SETUP.md` | Modal profile, volumes, GPU commands, cost/fair-play notes |

## Teammate reports (upstream)

| Report | Contents |
| :--- | :--- |
| `01_infrastructure_readiness.md` | Initial infrastructure readiness report |
| `02_cloud_and_auth_setup.md` | Cloud/Kaggle/GitHub Actions auth plan |
| `03_full_library_verification.md` | Library audit and template demos (teammate box) |
| `04_git_sync_and_deps_update.md` | Upstream sync + 72-package requirements update |

## Live / reference

| File | Contents |
| :--- | :--- |
| `AGENT_CONTEXT.md` | Shared context every subagent must read first |
| `LIVE_STATUS.md` | Auto-updated competition state (settings/journey/tasks) |
| `LOG.md` | Chronological operator log |
| `levels/INDEX.md` | Task index (auto-filled by `tools/intake.py` when unlocked) |
| `levels/BOARD.md` | Live task board (claims, plans, submissions, deadlines) |
| `runbooks/01..07_*.md` | Per-track CPU runbooks |
| `templates/TASK_BRIEF.md` | Per-task brief template |
| `templates/TASK_REPORT.md` | Per-task final report template |
| `templates/SUBMISSION_CHECKLIST.md` | Pre-submit gate |

> Reminder: `reports/07_RULES_AND_SCORING.md` says the **latest** submission per
> task counts for the team score — not the best. Never replace a good
> submission with a speculative one without reviewer approval.
