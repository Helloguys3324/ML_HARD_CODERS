# 10 — Upstream Sync & Local Revalidation

_Orchestrator, 2026-11-14 ~09:25 UTC._

## What changed on GitHub

Local `main` was at `3bc4a4a`; `origin/main` moved to `8428d3e`:

| Commit | Author | Summary |
| :--- | :--- | :--- |
| `b9a0058` | teammate | `docs: add full library and templates verification report` |
| `4e0946f` | Veaceslav Coltuc | `feat: upgrade LMML 2.0 domain readiness, CTF boundary, and requirements.txt` |
| `8428d3e` | teammate | `docs: report sync with upstream commit 4e0946f and dependencies update` |

`4e0946f` is large: **+7,670 / −380 lines** across templates (all 7),
`rust_core` (`graph_rl.rs`, `lib.rs`), `optimizations/fast_eda.py`,
`optimizations/free_api_client.py`, `optimizations/rust_fast_ext.py`,
`lmml_master.py`, `git_sync.py`, `AGENTS.md` (new CTF/untrusted-input section),
and a new `requirements.txt` (104 lines, 72 packages).

## Local sync performed

```bash
cd /home/kozak/mllm/MML
git add .gitignore reports tools
git commit -m "feat(agents): API intake/watch, CPU playbooks, war-room reports, baseline + git tooling"
git pull --rebase origin main      # clean, no conflicts
# HEAD: 6a5ddfb (local) on top of 8428d3e
```

Our local commit contains only `reports/`, `tools/` and a `work/` line in
`.gitignore`; no upstream files were overwritten. It has not been pushed yet
(push after revalidation).

## Local updates after the sync

1. **Rust core rebuilt** against new `graph_rl.rs` / `lib.rs`:
   `nix-shell -p rustc cargo gcc --run 'cargo build --release'` → OK;
   `cargo test --release` → **7/7 pass**.
2. **Python dependencies.** Installed missing CPU-compatible packages from the
   new `requirements.txt`: `torchaudio`, `torch-geometric`, `pennylane`,
   `pennylane-lightning`, `sb3-contrib`, `scikit-image`, `albumentations`,
   `faster-whisper`, `ctranslate2`, `av`, `adversarial-robustness-toolbox`,
   `nltk`, `seqeval`, `sacrebleu`, `rouge_score`, `rapidfuzz`, `tiktoken`,
   `cryptography`, `argon2-cffi`, `beautifulsoup4`, `lxml`, `orjson`,
   `tabulate`, `kaggle`, `kagglehub`, `onnxruntime`, `langchain`, `langgraph`,
   `openai`, `groq`, `google-genai`.
   - `lmml_master.py health` now: **10/10 LMML packages available**.
   - `bitsandbytes` was intentionally not installed (CUDA-only, no GPU here).
   - `torch` remains 2.15.0+cu132; the pinned `2.6.0+cu124` in
     `requirements.txt` is not appropriate for this box and was not downgraded.
3. **API client patch.** `optimizations/sigmoid_api.py` (used by
   `orchestrator.py`) now resolves tokens from `SIGMOID_TOKEN`,
   `SIGMOID_AUTH_TOKEN`, `/home/kozak/mllm/.sigmoid_token`, then the legacy
   secrets file; file upload now sends `fileSize`/`fileName`/`fileExtension`
   and finalizes with `fileStorageKey`. `tools/sigmoid_api.py` is unchanged and
   remains the canonical CLI.
4. **Post-sync self-tests** (all pass):
   - `rust_fast_ext.py --test`
   - `fast_eda.py --self-test`
   - `ctf_tactics.py --self-test`
   - `blender.py`
   - `math_tricks.py`
   - `cpp_fast_ext.py --self-test`
   - `sys_bench.py --kernels-only --quick --json`
   - `lmml_master.py test-templates --syntax-only` → **7/7**
5. **Full template demos after sync.** `lmml_master.py test-templates
   --timeout 300` → **6/7 pass**:

   | Template | Result | Runtime |
   | :--- | :--- | ---: |
   | `01_nlp_fast.py` | PASS (score 1.00000) | 42.4 s |
   | `02_tabular_fast.py` | PASS (AUC 0.93069) | 54.9 s |
   | `03_cv_fast.py` | PASS (acc 0.90625) | 20.7 s |
   | `04_audio_genai_fast.py` | PASS (composite 0.91667) | 19.1 s |
   | `05_adversarial_sec.py` | **FAIL seed-42 STRIP assert (AUC 0.5312)** | 12.5 s |
   | `06_rl_fast.py` | PASS (reward 500.0) | 18.3 s |
   | `07_frontier_gnn_pinn.py` | PASS (PINN RMSE 0.00346) | 21.9 s |

   `05_adversarial_sec.py --demo --seed 1` → **PASS** (score 0.9013, 4.5 s).
   The failure is a deterministic seed/data property of the synthetic STRIP
   test, not a Linux/NixOS issue. Use `--seed 1` for the demo or ignore the
   demo assertion in real tasks. **Do not edit the template to hide it.**

## Open issues

- ~~**Modal credentials incomplete.**~~ **Resolved:** the user supplied the
  `as-…` token secret. Token verified, profile `democracy1984` activated,
  `lmml-data`/`lmml-cache` volumes visible. GPU smoke test is running; see
  `reports/16_MODAL_SETUP.md`.
- **Tasks still locked.** `allowSubmissions=false`, `journey=null`,
  `GET /task` → 403. No task data yet.
- **Runbook/template drift.** `reports/runbooks/*` were written before the
  template rewrite; flag names and example commands need a consistency pass.
- **Security demo seed flake.** Deterministic at seed 42; `--seed 1` passes.
  Do not treat it as a broken environment.
- **`git_sync.py` non-dry path is not concurrency-safe.** The new `--dry-run`
  is read-only (verified by agent L in a clone), but a real run still does
  `git add -A`, commits and pushes without fetch/rebase. Use
  `tools/report_commit.py` for report/tool commits; use `git_sync.py` only at a
  quiet checkpoint when the scoreboard must be regenerated.
