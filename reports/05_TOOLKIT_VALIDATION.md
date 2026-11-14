# 05 — Toolkit Validation (`optimizations/` + `rust_core/`)

**Verdict: PASS** — every mandatory self-test passed; all four Rust CLI subcommands work on
synthetic data; Python EDA/CTF/blending/math utilities run correctly on Linux/NixOS with no
code changes required. Two non-blocking edge-case defects and several usage limitations are
documented below.

## 0. Environment

| Item | Value |
|---|---|
| Host | Linux/NixOS, 4 logical CPU (2 physical), 15.5 GB RAM, **no CUDA** |
| Python | 3.13.17 via `/home/kozak/mllm/py` (sets NixOS `LD_LIBRARY_PATH`) |
| Stack | numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, polars 2.1.0, lightgbm 4.7.0, numba 0.68.1 |
| Rust | prebuilt `rust_core/target/release/lmml_rs` + `liblmml_rust.so` (not rebuilt) |
| Threads | `safe_threads = 4`; Blender uses `_safe_n_jobs() == 4` |
| Artifacts | `work/toolkit/` (all synthetic data, harnesses, JSON outputs) |

All commands below were run from `/home/kozak/mllm/MML` with the repo wrapper.

## 1. Self-tests

| Tool | Command | Status | Wall time | Notes |
|---|---|---|---|---|
| Rust FFI | `/home/kozak/mllm/py optimizations/rust_fast_ext.py --test` | **PASS** | 4.15 s | Rust C-ABI + NumPy fallback parity, exit 0 |
| Fast EDA | `/home/kozak/mllm/py optimizations/fast_eda.py --self-test` | **PASS** | 6.21 s | exit 0 |
| CTF tactics | `/home/kozak/mllm/py optimizations/ctf_tactics.py --self-test` | **PASS** | 1.17 s | 9 files, 23 CRITICAL / 4 WARN / 4 INFO in fixture; exit 0 |
| Blender | `/home/kozak/mllm/py optimizations/blender.py` | **PASS** | 91.93 s | No CLI/argparse: invokes unit suite on `__main__` |
| Math tricks | `/home/kozak/mllm/py optimizations/math_tricks.py` | **PASS** | 5.68 s | No CLI/argparse: invokes self-test on `__main__` |
| C++ ext | `/home/kozak/mllm/py optimizations/cpp_fast_ext.py --self-test` | **PASS** | 6.87 s | Built/cached under `/tmp/lmml2026_cpp_ext_cache` |
| Sys bench | `/home/kozak/mllm/py optimizations/sys_bench.py --kernels-only --quick --json` | **PASS** | 23.52 s | exit 0; JSON copied to `work/toolkit/sysbench_kernels_quick.json` |

**Important:** `blender.py --help` and `math_tricks.py --help` do **not** print help — both
scripts run their full self-test instead (`blender.py` takes ~92 s). Treat them as libraries;
use `--help` only on `fast_eda.py`, `ctf_tactics.py`, `rust_fast_ext.py`, `cpp_fast_ext.py`,
`sys_bench.py`.

No-Rust fallback was also exercised (`LMML_NO_RUST=1` / `--no-rust`):

| Check | Result |
|---|---|
| `fast_eda.py --data work/toolkit/binary --no-rust --json` | exit 0, `rust_used=false`, same drift/leak findings; 5.87 s |
| `ctf_tactics.py --data work/toolkit/ctf_sample --no-rust --json` | exit 0, `rust_used=false`, same 5 CRITICAL / 1 WARN findings; 0.93 s |

## 2. Rust CLI (`rust_core/target/release/lmml_rs`)

### 2.1 `eda`

| Command | Exit | Time | Sanity |
|---|---|---|---|
| `lmml_rs eda work/toolkit/binary/train.csv --json` | 0 | 31 ms | rows=1200, cols=8, null %, `id_suspects=[row_id, leak_id]` |
| `lmml_rs eda work/toolkit/multiclass/train.csv --json` | 0 | 29 ms | rows=900, cols=7 |
| `lmml_rs eda work/toolkit/regression/train.csv --json` | 0 | 31 ms | rows=1000, cols=7 |
| `lmml_rs eda train.csv --max-rows 100 --json` | 0 | 31 ms | `rows=1200`, `sampled_rows=100` (correct sampling) |
| `lmml_rs eda /nonexistent.csv --json` | **1** | — | stderr `[ERROR] Cannot stat ...` |
| `lmml_rs eda /tmp/sub_empty.csv --json` | **1** | — | stderr `[ERROR] Empty CSV ...` |
| BOM + CRLF + UTF-8 + quoted commas test (`/tmp/edge.csv`) | 0 | 31 ms | Correct: 3 rows, unicode header, null counts; ragged/blank lines tolerated |

### 2.2 `validate`

All tests compared `/tmp/sub_*.csv` against `work/toolkit/binary/sample_submission.csv`
(400 rows, columns `row_id,target`).

| Case | Exit | `ok` | Errors/warnings |
|---|---|---|---|
| Correct aligned submission | 0 | true | warning only if prediction column constant |
| Variable predictions (`/tmp/sub_var.csv`) | 0 | true | none |
| Wrong column (`pred`) | **1** | false | `Column mismatch ...` |
| 395 rows instead of 400 | **1** | false | row count + ID alignment mismatch |
| One `NaN` cell | **1** | false | `Found 1 null/NaN/empty cells` |
| One `+Inf` cell | **1** | false | `Found 1 Inf/-Inf cells` |
| One `-Inf` cell | **1** | false | `Found 1 Inf/-Inf cells` |
| Shuffled IDs (same set) | **1** | false | `ID alignment mismatch on column 'row_id'` |
| Constant predictions | 0 | true | warning `Constant prediction column(s)` |
| Header-only CSV (0 data rows) | **1** | false | `Submission CSV has 0 data rows` |
| 0-byte CSV | **1** | false | `Submission file is 0 bytes` |
| Ragged CSV (`a,b` then `1,2` then `3`) | **1** | false | `1 rows have mismatched column count` |
| No sample CSV supplied (`validate sub.csv --json`) | 0 | true | intrinsic checks only |

Note: `--allow-constant` does **not** change the JSON warning text; constant columns are
already treated as a warning (exit 0) by default. This is a no-op flag, not a functional bug.

### 2.3 `ctf-scan`

Fixture: `work/toolkit/ctf_sample/` (plain flag, pickle RCE, zip-comment flag + zip-slip,
tar.gz secret, PNG trailer flag).

| Command | Exit | Time (internal) | Findings |
|---|---|---|---|
| `lmml_rs ctf-scan work/toolkit/ctf_sample --json` | 0 | 0.196 ms | 5 files; 3× `flag_or_secret` (txt, zip comment, png tail), 1× `image_trailing` |
| `lmml_rs ctf-scan work/toolkit/ctf_sample/evil.pkl --json` | 0 | 0.059 ms | **0 findings** — CLI does not audit pickle opcodes |
| `lmml_rs ctf-scan work/toolkit/ctf_sample/nested.tar.gz --json` | 0 | 0.063 ms | **0 findings** — CLI does not decompress archives |
| `lmml_rs ctf-scan /nope --json` | **1** | — | stderr `[ERROR] Target not found: /nope` |

For deep forensics use `ctf_tactics.py` (see §4): it detects the pickle RCE, zip-slip and
zip comment; neither tool expands `.tar.gz` member contents (workaround: extract first).

### 2.4 `bench`

| Command | Exit | Time | Notes |
|---|---|---|---|
| `lmml_rs bench --n 100000 --json` | 0 | 310 ms | 14 kernels, `threads=4`, plausible results (`roc_auc=1.0`, `logloss=0.2335`, `max_entropy=8.0`, …) |
| `lmml_rs bench --n 0 --json` | **134 (SIGABRT)** | — | **Bug:** panic `index out of bounds: len is 0, index 18446744073709551615` at `rust_core/src/cli.rs:832` |

Suggested one-line fix (not applied — Rust binary is shared and already built):
`let n = n.max(1);` at the top of `cmd_bench`, or reject `n == 0` in the `"bench"` arg parser.

## 3. Synthetic tabular suite

Generator: `work/toolkit/make_synthetic.py` (seed-fixed, regenerable).

| Dataset | Rows train/test | Shape | Injected pathologies |
|---|---|---|---|
| `work/toolkit/binary` | 1200 / 400 | 8 / 7 cols | 13.25 % positive; 5–11 % MCAR nulls; categoricals; `leak_id` ID leak; `Oslo`/`enterprise` unseen in test; distribution shift |
| `work/toolkit/multiclass` | 900 / 300 | 7 / 6 cols | 3 classes (343/314/243); nulls; categoricals; unseen categories; mild shift |
| `work/toolkit/regression` | 1000 / 350 | 7 / 6 cols | continuous right-skewed target (skew 2.23); nulls; `income` near-perfect leak; severe shift |
| `work/toolkit/imbalance` | 2000 / 500 | 4 / 3 cols | 3.65 % positive minority |

### 3.1 `fast_eda.py` results

Command: `/home/kozak/mllm/py optimizations/fast_eda.py --data work/toolkit/<dataset> --json`

| Dataset | Exit | Wall (incl. imports) | Profiler | Detected |
|---|---|---|---|---|
| binary | 0 | 7.89 s | 0.80 s | `task_type=binary`, target `target`, minority 13.25 %, metric `roc_auc or logloss`; `id_suspects=[row_id, leak_id]`; drift **severe** (`adv_auc=0.8482`, top `city,plan,score,age,income`); unseen `city`/`plan` |
| multiclass | 0 | 5.95 s | 0.57 s | `task_type=multiclass`, 3 classes, imbalance ratio 1.41, metric `multi_logloss or accuracy`; drift **mild** (`adv_auc=0.6766`); unseen `city`/`plan` |
| regression | 0 | 6.32 s | 0.43 s | `task_type=regression`, skew 2.23, metric `rmsle or rmse (consider log1p)`; **`NEAR-PERFECT LEAK: income r=+1.0000`**; drift **severe** (`adv_auc=0.7715`) |
| imbalance | 0 | ~6 s | — | minority 3.65 % → `pr_auc / roc_auc (severe class imbalance)` + threshold/focal-loss note |

Notes:
- Imbalance warning is intentionally thresholded at **<5 %** for binary; 13.25 % correctly
  only changes the reported minority ratio, not the metric.
- `leak_id` is surfaced under `id_suspects` (string IDs cannot be Pearson-correlated); review
  `id_suspects` manually — no explicit `LEAK ALERT` is emitted for string ID leaks.
- Drift `adv_auc` is computed by the Python/sklearn path even when Rust EDA is enabled;
  `rust_used=true` confirms the Rust CSV profile was merged in.

### 3.2 `ctf_tactics.py` results

| Target | Exit | Wall | Summary |
|---|---|---|---|
| `work/toolkit/binary` | 0 | 1.04 s | 3 files, 0 findings (expected) |
| `work/toolkit/multiclass` | 0 | 1.00 s | 3 files, 0 findings |
| `work/toolkit/regression` | 0 | 0.97 s | 3 files, 0 findings |
| `work/toolkit/ctf_sample` | 0 | 0.94 s | 5 files, **8 CRITICAL / 1 WARN / 1 INFO**: `pickle_rce_opcodes` (`STACK_GLOBAL(os.system)`), zip comment flag, zip-slip, PNG trailing bytes, plain flag, `flag_leak` |

Tar member contents are not inspected by either scanner. Use `tar -xzf` + rescan when a
`*.tar.gz` is present.

## 4. `blender.py` + `math_tricks.py` on synthetic OOF predictions

Harness: `work/toolkit/bench_toolkit.py` — 6 binary models (`n=4000`), 4 multiclass models
(4 classes, `n=1333`), 4 regression models (`n=2000`), 4-thread CPU.
Raw JSON: `work/toolkit/bench_toolkit_n4000.json`.

| Function | Time | Sanity result |
|---|---|---|
| `blender.caruana_hill_climb` (AUC, 6 models, 2 bags) | 7.4 ms | weights sum 1.0, picked strongest model; synthetic OOF AUC 1.0 (separable fixture) |
| `blender.optimize_weights_nelder_mead` (logloss, 2 restarts) | 715 ms | logloss 0.2199, weights sum 1.0 |
| `blender.optimize_weights_dirichlet` (200 samples) | 56 ms | logloss 0.2800 |
| `blender.rank_average` | 3.8 ms | AUC 0.926 |
| `blender.blend_predictions` (`prob`/`rank`/`log_odds`/`power_rank`/`probit_rank`) | 0.1 / 3.6 / 0.7 / 3.1 / 5.3 ms | AUC 0.9635 / 0.9260 / 0.9613 / 0.9260 / 0.9189 |
| `blender.oof_nonneg_stacker` (classification, 5 folds) | 573 ms | OOF AUC 0.9999, non-negative weights, test shape 1000 |
| `blender.oof_ridge_stacker` (regression, 5 folds) | 11.0 ms | OOF RMSE 0.2449, test shape 1000 |
| `blender.ensemble_diversity_report` | 19.4 ms | effective models 1.217, mean corr 0.879 |
| `blender.multi_seed_ensemble_stability` (2 seeds × 3 folds) | 79.5 ms | CV mean 1.0, std 0.0 (synthetic) |
| `blender.auto_blend` (2 modes, `n_splits=2`) | **3.60 s** | 5 candidates, winner `caruana_nested`, score 1.0 |
| `blender.caruana_hill_climb` multiclass (logloss) | 168 ms | logloss 0.5137 |
| `blender.adversarial_validation` (3000×8, 3 folds) | **75.07 s** | AUC 0.8715, severity `severe`, sample weights returned |

`blender._rust_ready() == True`, `_safe_n_jobs() == 4`.

| `math_tricks` function | Time | Sanity result |
|---|---|---|
| `optimal_threshold_sweep` (F1) | 0.51 ms | threshold 0.5103, F1 0.9802 |
| `oof_optimal_threshold` (5-fold) | 16.6 ms | median threshold 0.5103, nested F1 0.979 |
| `expected_f1_optimal_threshold` | 0.36 ms | threshold 0.3811 |
| `calibration_metrics` (raw) | 2.7 ms | ECE 0.2237, MCE 0.3313, Brier 0.0723, LogLoss 0.2948 |
| `fit/apply_temperature_scaling` (binary) | 36.9 ms | T=0.1476, ECE 0.2237 → **0.0034** |
| `calibrate_oof_and_test` isotonic / platt / temperature | 30.9 / 497 / 325 ms | OOF ECE 0.0077 / 0.0045 / 0.0045, test preds shape 1000 |
| `SplitConformalRegressor` (α=0.1) | not separately timed | coverage 0.893, mean width 1.169, `q_hat=0.5846` |
| `cqr_intervals` (α=0.1) | 0.11 ms | coverage 0.893, mean width 1.169 |
| `ConformalClassifier` LAC / APS / RAPS (α=0.1) | 0.25 / 7.1 / 1.1 ms | coverage 0.899 / 0.904 / 0.904; avg set size 2.36 / 2.47 / 2.47 |
| `MahalanobisOOD` (1000×16 in, 200 out) | 168 ms | in-vs-out rank AUC-like 1.0 |
| `adversarial_validation_and_weights` (2000×10) | 243 ms | AUC 0.8579, top feature 4, severe |
| `rank_gauss` (2000 values) | 0.19 ms | mean 0.0, std 0.9967 |
| `oof_target_encode` (5000 rows, 50 cats) | 29.6 ms | shapes (5000,), (50,) |

Conformal coverage is slightly below nominal (0.893 vs 0.9) at n=2000 calibration
samples — expected finite-sample variation, not a defect.

## 5. `sys_bench.py` backend matrix

Command: `/home/kozak/mllm/py optimizations/sys_bench.py --kernels-only --quick --json`
(23.52 s wall). Backend availability: **Rust ✓, C++ ✓, Numba ✓, NumPy ✓, sklearn ✓, CUDA ✗**.
No required library is missing.

| Kernel | n | sklearn | NumPy | Numba | C++ | Rust |
|---|---|---|---|---|---|---|
| ROC-AUC | 80 000 | 62.19 ms | 41.00 ms | 30.33 ms | 15.71 ms | **11.01 ms** |
| LogLoss | 80 000 | 58.17 ms | 8.23 ms | — (not exposed) | 1.97 ms | **1.95 ms** |
| Floyd–Warshall | 72 | — | — | **0.40 ms** | 0.38 ms | — |
| RankGauss | 40 000 | — | scipy 12.34 ms | — | — | **5.97 ms** |
| Hill-climb | 40 000 × 5 | — | python 85.66 ms | — | — | **58.33 ms** |

`lmml_rs bench --n 80000 --json` (threads=4) exercised all 14 kernels successfully:
`roc_auc 20.6`, `logloss 7.3`, `macro_f1 0.37`, `rmse 0.63`, `ndcg_at_50 6.3`,
`rolling_stats 16.7`, `oof_target_encode 17.7`, `rank_gauss 15.0`, `hill_climb 84.7`,
`hash_ngrams 2.0`, `dijkstra 0.55`, `bfs 6.4`, `gae 0.43`, `byte_entropy 0.34` ms.

Environment caveat: `sys_bench` reports `cpu_physical=2`, `cpu_logical=4`, `safe_threads=4`
(`nproc` also returns 4). Treat all timings as **4-logical-thread** measurements.

## 6. Failures, limitations, fixes

1. **`lmml_rs bench --n 0` panics** (exit 134, index-out-of-bounds at `cli.rs:832`).
   Non-blocking for normal `n ≥ 1`. One-line fix: clamp `n` (`let n = n.max(1);`) or
   return a usage error. Not applied here to avoid changing the shared prebuilt binary.
2. **`validate --allow-constant` is a no-op**; constant columns are always a warning and
   exit 0. Remove the flag or make it suppress the warning.
3. **`lmml_rs ctf-scan` is a shallow byte scanner** (flags/secrets, magic mismatch, trailing
   data, unicode). It misses pickle RCE opcodes and archive members. Use
   `ctf_tactics.py` for deep forensics; extract `.tar.gz` manually first.
4. **`blender.adversarial_validation` is LightGBM-bound**: 75 s for 3000×8 with 3 folds.
   Prefer `fast_eda.py` drift (logistic, sub-second) for triage, or
   `math_tricks.adversarial_validation_and_weights` (243 ms at 2000×10) when importance
   weights are not required.
5. **`blender.py` / `math_tricks.py` have no CLI parser**: `--help` executes the unit suite
   (92 s / 6 s). Import them as libraries in challenge scripts.
6. **String-ID leakage is only listed, not alerted**: review `fast_eda`'s `id_suspects`
   manually; numeric ID leaks do trigger `LEAK ALERT`.
7. No self-test, import, FFI, C++ build, or Numba failures occurred. `LMML_NO_RUST=1`
   parity paths work but are slower on EDA/CTF.

## 7. Cheat sheet — “use this when”

| Situation | Use | Command / call |
|---|---|---|
| First look at a challenge folder | `fast_eda.py` | `/home/kozak/mllm/py optimizations/fast_eda.py --data <dir> --json` (6–8 s) |
| Sub-second raw CSV profile / CI schema check | `lmml_rs eda` | `rust_core/target/release/lmml_rs eda <csv> --json` (~30 ms/1k rows) |
| Pre-submit gate (shape/IDs/NaN/Inf/constant) | `lmml_rs validate` | `... validate submission.csv sample_submission.csv --json`; rely on exit code 1 |
| Forensics / stego / pickle / zip / PDF / WAV / safetensors / prompt injection | `ctf_tactics.py` | `/home/kozak/mllm/py optimizations/ctf_tactics.py --data <dir> --json` (~1 s) |
| Bulk byte/flag scan of a huge directory | `lmml_rs ctf-scan` | `... ctf-scan <dir> --json` (raw bytes only; no archive expansion) |
| Pick ensemble weights / leaderboard selection | `blender.auto_blend` | `auto_blend(oof, y, metric="auc", task="classification")` |
| Quick deterministic blend | `blender.caruana_hill_climb` | `caruana_hill_climb(oof, y, metric="auc")` (ms) |
| Meta-stacking | `blender.oof_nonneg_stacker` | classification/regression, nested 5-fold, non-negative weights |
| Train/test shift + importance weights | `math_tricks.adversarial_validation_and_weights` | fast (~0.25 s); use `blender.adversarial_validation` only if you need its detailed LightGBM report |
| Probability calibration | `math_tricks.calibrate_oof_and_test` | `method="isotonic"` (fast) or `"temperature"` for multiclass |
| Threshold tuning | `math_tricks.oof_optimal_threshold` | leak-free nested threshold, `metric="f1"/"mcc"/...` |
| Prediction intervals / sets | `math_tricks` conformal | `SplitConformalRegressor`, `cqr_intervals`, `ConformalClassifier(method="raps")` |
| CPU kernel performance sanity | `sys_bench.py` | `/home/kozak/mllm/py optimizations/sys_bench.py --kernels-only --quick --json` |
| Rust FFI correctness check | `rust_fast_ext.py` | `--test` (4 s) or `LMML_NO_RUST=1` for parity fallback |

## 8. Reproduction artifacts

| Path | Purpose |
|---|---|
| `work/toolkit/make_synthetic.py` | regenerates binary/multiclass/regression/imbalance datasets |
| `work/toolkit/bench_toolkit.py` | blender/math_tricks synthetic OOF benchmark |
| `work/toolkit/bench_toolkit_n4000.json` | raw benchmark timings/results |
| `work/toolkit/sysbench_kernels_quick.json` | raw sys_bench output |
| `work/toolkit/lmml_rs_bench_n100k.json` | Rust CLI bench output |
| `work/toolkit/lmml_rs_ctf_scan.json` | Rust CLI ctf-scan output |
| `work/toolkit/{binary,multiclass,regression,imbalance}/` | synthetic CSVs |
| `work/toolkit/ctf_sample/` | forensic fixture (pickle/zip/tar/png/flags) |

No source code was modified; no commit/push was performed; no tokens were touched.
