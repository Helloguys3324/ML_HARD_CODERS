# CPU-Only Playbook — LMML 2026 (4 cores, 15 GB RAM, no GPU)

> Scope: 24-hour ML/AI marathon, each task budgeted at **≤120 minutes**.
> Machine: Linux/NixOS, **4 logical cores**, ~15 GB RAM, **no CUDA**. Other agents
> share the same box, so wall-clock is 2–4× slower while they run.
> Use `/home/kozak/mllm/py` for everything; never call `venv/bin/python` directly.
> Fair-play default is **no pretrained weights, no external data, no external APIs**.

Deeper background (read before improvising): `pro_playbooks/02_pro_feature_engineering.md`
(tabular FE/TE), `04_pro_nlp_and_llm_hacks.md` (TF-IDF stack), `03_pro_cv_and_vision_tricks.md`
(CV pitfalls), `05_pro_audio_and_genai_hacks.md` (audio), `06_pro_adversarial_and_ctf_redteam.md`
(sec), `07_pro_rl_and_control_hacks.md` (RL), `08_pro_frontier_gnn_pinn_qml.md` (frontier),
`09_pro_loss_metrics_and_tuning.md` (post-processing), `10_pro_24h_warroom_workflow.md`
(war-room cadence). This playbook overrides their GPU/Modal assumptions for this box.
Per-track runbooks: `reports/runbooks/01_nlp.md`, `02_tabular.md`, `03_cv.md`,
`04_audio_genai.md`, `05_sec.md`, `06_rl.md`, `07_frontier.md`.

---

## 0. Hard truths first

1. **CPU-only is competitive on tabular, classical NLP, graph feature engineering,
   forensics, exact/heuristic RL, and small audio probes.** It is *not* competitive
   on transformer fine-tuning, large vision training, Whisper, or video.
2. On this box a from-scratch `resnet18` at 224 px trains at roughly **1–2 images/s**
   under load. A 1k-image, 5-fold, 3-epoch job is >2 hours. Do not start it.
3. A `LightGBM` 50k×40, 100-round model fits in **~0.5–2 s**; 20k text documents hash
   into 131k features in **~2 s** with Rust. This is where CPU wins.
4. **One heavy job at a time.** Parallel subagents already saturate 4 cores; two
   trainings double both runtimes and risk OOM. Check `uptime` before launching.
5. GPU-only work (DeBERTa/ModernBERT fine-tune, ViT/ConvNeXt 384+, YOLO training,
   Whisper-large, diffusion, >16-qubit statevector) gets a **format-valid baseline**
   and is then abandoned, not "optimized".

---

## 1. Task intake triage (T+0 to T+10 min)

### 1.1 Fetch and preserve the task statement

```bash
export PY=/home/kozak/mllm/py
export REPO=/home/kozak/mllm/MML
cd "$REPO"

mkdir -p work/tasks/<task_id>
$PY tools/sigmoid_api.py task <task_id>        > work/tasks/<task_id>/task.json
$PY tools/sigmoid_api.py task-overview <task_id> > work/tasks/<task_id>/overview.json 2>/dev/null || true
```

Data usually lands in `lvl_XX/data/` (the `lmml_master.py new` scaffold). If the
task payload contains attachment URLs, download them there and keep the raw bytes
untouched. Never commit `work/` or `lvl_XX/data/`.

### 1.2 Decision tree + modality table

```text
submissionType == "text" and statement mentions flag/CTF
  -> sec: run ctf_tactics + lmml_rs ctf-scan, submit exact text
submissionType == "prompt" or chat/LLM evaluation
  -> genai: RAG/prompt audit; external APIs only if explicitly allowed
else (file submission)
  inventory data/
  ├── .wav/.mp3/.flac or audio path column        -> audio
  ├── image files/folders or image path column    -> cv
  ├── src,dst[,weight] edges / nodes / QUBO / PDE -> frontier
  ├── .pkl/.pt/.safetensors/.zip/.pdf/.bin, model audit, poison/backdoor -> sec
  ├── gym env / actions / rewards / scheduling    -> rl
  ├── long text column + label                    -> nlp (TF-IDF first)
  └── CSV/Parquet with tabular target             -> tabular
```

If two tracks fit, choose the one with the **cheaper valid baseline** (tabular >
nlp > audio > cv > frontier > rl), ship it, then revisit.

| Signal in task JSON / data dir | Track | Template |
| :--- | :--- | :--- |
| CSV/Parquet, one target column, mixed numeric/categorical | `tabular` | `templates/02_tabular_fast.py` |
| Text column (`text`, `comment`, `review`, `prompt`) + label | `nlp` | `templates/01_nlp_fast.py` |
| Image files/folders or `image` path column | `cv` | `templates/03_cv_fast.py` |
| `.wav/.mp3/.flac` or `audio` path column | `audio` | `templates/04_audio_genai_fast.py` |
| `.pkl/.pt/.safetensors/.zip/.pdf/.bin`, flags, prompts, model audit | `sec` | `templates/05_adversarial_sec.py` |
| `gym`/environment, actions, rewards, scheduling, routing | `rl` | `templates/06_rl_fast.py` |
| Edge/node list, PDE, QUBO matrix, graph labels | `frontier` | `templates/07_frontier_gnn_pinn.py` |
| `submissionType: prompt` / chat evaluation | `genai` | `04_audio_genai_fast.py --mode genai_rag` |
| `submissionType: text` + "flag" | `sec` / CTF | `tools/sigmoid_api.py submit-text` |

Ambiguous cases:
- CSV with a long text column **and** metadata → start NLP (TF-IDF), add tabular
  metadata as side features; do not start with a transformer.
- CSV of image paths → CV, not tabular. CSV of audio paths → audio.
- CSV of `src,dst,weight` → frontier graph, not tabular.
- "Detect poisoned samples / backdoored class / adversarial example" → sec, even
  if the input is a CSV or `.npz`.

### 1.3 Metric mapping (read the exact wording, do not guess)

| Metric words | Predict | Post-process |
| :--- | :--- | :--- |
| AUC / ROC / Gini | probability or score | rank-average blend; no threshold |
| LogLoss / Brier | probability | clip `[1e-6, 1-1e-6]`, temperature/Platt calibration |
| Accuracy / F1 / MCC / balanced accuracy | class or probability | OOF threshold sweep, class priors |
| RMSE / MAE / R² | real number | no sigmoid; clip to train range |
| RMSLE | non-negative | `log1p` target, `expm1` prediction |
| QWK / ordinal | integer/float | regression + boundary search |
| mAP / NDCG / recall@k | ranked list/score | rank blend, cut at k |
| WER / CER / BLEU | text | normalize case/punctuation; exact scorer |
| ASR / attack success | adversarial sample + preds | attack only as many samples as budget allows |
| IoU / Dice | mask/boxes | threshold sweep, TTA |
| Energy / cost (minimize) | assignment/vector | exact solver first |

### 1.4 Fair-play gate (default deny)

Record the exact permission sentence in `work/tasks/<task_id>/fairplay.md` before
enabling anything:

- **Pretrained models**: allowed only if the statement says "pre-trained",
  "foundation model", or "open-source model weights" are permitted. Otherwise use
  classical features; frozen *random* features are useless, so skip them.
- **External data**: allowed only if explicitly stated. No Kaggle/HF datasets, no
  scraping, no synthetic data from an external model.
- **External APIs**: allowed only if explicitly stated. `free_api_client.py` stays
  behind `--allow-external-api`.
- Task-provided train/test/hints are always allowed. Transductive use of unlabeled
  test features is allowed unless the rules forbid it; never use test labels.
- No cross-team flags, weights, or solutions. No public-leaderboard probing.

If the statement is silent, **assume forbidden** and note the assumption in the
task report.

| Template | Fair-play gate | Default |
| :--- | :--- | :--- |
| `01_nlp_fast.py` | `--pretrained` | off |
| `03_cv_fast.py` | `--pretrained` | off |
| `04_audio_genai_fast.py` | `--pretrained`, `--allow-external-api` | off |
| `05_adversarial_sec.py` | `--pretrained`, `--allow-external-api` (`store_true`) | off |
| `02`, `06`, `07` | none (classical/scratch only) | n/a |

### 1.5 Reusable intake script

Save as `work/tasks/intake.py` (do not edit repo code):

```python
import json, sys
from collections import Counter
from pathlib import Path
import polars as pl

root = Path(sys.argv[1])                 # dir containing task.json
data = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "data"
task = json.loads((root / "task.json").read_text())

def walk(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            print(f"{prefix}{k}: {type(v).__name__}")
            if isinstance(v, (dict, list)):
                walk(v, prefix + "  ")
    elif isinstance(obj, list) and obj:
        walk(obj[0], prefix + "  [0].")

print("== TASK KEYS =="); walk(task)
blob = json.dumps(task).lower()
for keys, metric in [
    (("roc_auc", "auc", "gini"), "AUC"),
    (("logloss", "log loss", "brier"), "LogLoss"),
    (("macro f1", "f1", "mcc"), "F1/MCC"),
    (("rmse", "mae", "rmsle", "r2"), "Regression"),
    (("accuracy", "exact match"), "Accuracy"),
    (("wer", "cer", "bleu"), "Text generation"),
    (("ndcg", "map@", "recall@"), "Ranking"),
]:
    if any(k in blob for k in keys):
        print("== METRIC GUESS ==", metric)
files = [p for p in data.rglob("*") if p.is_file()]
print("== FILE TYPES ==", Counter(p.suffix.lower() for p in files).most_common(12))
print("== SIZE (MB) ==", round(sum(p.stat().st_size for p in files) / 1e6, 2))

for p in files:
    if p.suffix.lower() in {".csv", ".parquet"}:
        df = pl.read_csv(p, n_rows=5000) if p.suffix == ".csv" else pl.read_parquet(p)
        print(f"\n{p.name}: sample={df.shape} cols={df.columns}")
        for c in df.columns:
            n = df[c].n_unique()
            if n <= 20 or "id" in c.lower() or "target" in c.lower() or "label" in c.lower():
                print(f"  {c}: dtype={df[c].dtype} nunique={n} nulls={df[c].null_count()}")
```

Run it and read the output, not just the first lines:

```bash
$PY work/tasks/intake.py work/tasks/<task_id> lvl_XX/data
```

Then run the two forensic scanners before modeling:

```bash
$PY optimizations/fast_eda.py --data lvl_XX/data
$PY optimizations/ctf_tactics.py --data lvl_XX/data
./rust_core/target/release/lmml_rs eda lvl_XX/data/train.csv --json
./rust_core/target/release/lmml_rs ctf-scan lvl_XX/data --json
```

`fast_eda.py` reports target balance, ID/target leakage, train/test overlap,
unseen categories, and an adversarial AUC. `ctf_tactics.py` reports hidden flags,
polyglots, EXIF/LSB stego, pickle opcodes, and prompt injections. Both are cheap
(seconds) and have found real signals in ML-CTFs.

---

## 2. Time-boxed workflow per task (120 min hard stop)

| Window | Goal | Exit criterion |
| :--- | :--- | :--- |
| **T+0–10** | Intake, fair-play gate, EDA, CTF scan, scaffold | Track + metric + submission schema written down |
| **T+10–20** | **Fastest valid baseline** | `submission.csv` passes `lmml_master.py check`; uploaded to portal |
| **T+20–50** | One improvement with the best ROI (features > model) | OOF score improves and is reproducible |
| **T+50–70** | Validation: CV scheme, adversarial AUC, calibration/threshold | Nested OOF check; no leakage |
| **T+70–100** | Second model family + blend/ensemble | Blend beats best single on OOF |
| **T+100–115** | Freeze, re-run best config end-to-end, `check` | Same score ± noise, file hash recorded |
| **T+115–120** | Submit final + write `reports/levels/lvl_XX.md` | Portal confirms; report has command + score |

The practical 10/30/30/20 split: **10 min intake + first baseline**, **30 min
highest-ROI improvement**, **30 min validation/ensemble**, **20 min freeze/submit**.
The table above only shifts the baseline submission to T+20 so it can be validated
first.

Rules:
- **Submit at T+20 even if the score is bad.** A valid baseline unlocks information
  and protects against a late failure. Keep the previous file as fallback.
- **45-minute sunk-cost breaker.** If the baseline is not producing a valid file by
  T+45, stop modeling and fix the pipeline/format.
- On the 24-hour clock, do breadth first: one baseline per task, then return to the
  highest `points_gap / effort` tasks. Do not spend 4 hours on one track.
- Every experiment gets a tag and its OOF/test arrays saved. Never overwrite the
  previous best submission.

### 2.1 24-hour macro schedule

| Window | Phase | Action |
| :--- | :--- | :--- |
| H0–H2 | Unlock | Baseline every available task; validate submission schema; record scores |
| H2–H10 | Depth | Improve top-ROI tasks; one heavy job at a time; save OOF artifacts |
| H10–H18 | Breadth/ensembles | Second model family per task; blend; revisit failed tasks |
| H18–H22 | Stretch | Hard tasks (CV/audio/frontier) only if a cheap CPU path exists |
| H22–H24 | Freeze | Re-run best configs, `check`, final portal submissions, reports |

---

## 3. CPU budget table (4 cores; shared box, conservative)

| Track | 10 min | 30 min | 60 min | 120 min |
| :--- | :--- | :--- | :--- | :--- |
| **NLP** | TF-IDF word(1-2) + LogisticRegression/Ridge on ≤50k docs | + char(3-5), SVD(64), text meta, 5-fold | Full TF-IDF + Rust hash + LGB stack + calibration | Frozen MiniLM/BGE embeddings (only if pretrained allowed) + linear probe, or 2-family blend |
| **Tabular** | LightGBM 3-fold, 300–500 trees | 5-fold LGB + XGB, basic FE | + CatBoost, OOF TE, group/ratio features, 3-model blend | + MLP/Ridge diversity, 2 seeds, calibration, adversarial pruning |
| **CV** | Color/size/edge features + LGB | Tiny CNN 96 px, 2 folds, 2 epochs (≤2k images) | 128 px, 3 folds, 3 epochs, hflip TTA | Frozen pretrained embeddings + Ridge/LGB (only if allowed), or 96 px scratch + classical blend |
| **Audio/GenAI** | Log-mel stats + LGB on ≤1k clips; BM25 RAG | Conformer/ResNet1D 2 folds, 3 epochs (≤1.5k clips) | 3 folds, 8 epochs (≤2k clips) + spectral blend | 2 architectures + TTA + blend (≤3k clips); ASR only for short audio |
| **Sec** | `ctf_tactics.py` + `lmml_rs ctf-scan` + prompt audit | Small PGD/MI-FGSM on ≤100 samples; poison filter | Backdoor/Neural Cleanse on small model; 500-sample attack | Full evasion suite on ≤1k samples + ensemble surrogate |
| **RL** | Triage: exact MILP/DP/beam search | PPO 20k–50k steps, 4 envs, MLP | PPO 100k steps + action masking + eval checkpoint | 300k–500k steps or multi-seed ensemble (simple env only) |
| **Frontier** | Topological features + LGB; exact MILP for QUBO ≤18 vars | PINN (200 Adam + 25 LBFGS) or GNN 100 graphs | GNN 3 folds on ≤1k graphs + LGB on embeddings | + RWSE/LapPE, virtual node, QUBO parallel tempering, blend |

Estimates assume no other heavy job. Under current multi-agent load, multiply by
2–4×. Re-measure with `$PY lmml_master.py bench` and `uptime` before committing.

---

## 4. Exact entry points and commands

All commands run from `/home/kozak/mllm/MML`.

### 4.1 Health and track matrix

```bash
$PY lmml_master.py health          # cores, Rust, CUDA, caches
$PY lmml_master.py tracks          # continent -> template mapping
$PY lmml_master.py bench           # CPU/thread benchmarks
```

### 4.2 Scaffold a task

```bash
$PY lmml_master.py new lvl_05 --track tabular
# creates lvl_05/{solve.py,config.json,data/,submissions/,.vault/}
```

Track names: `nlp`, `tabular`, `cv`, `audio`, `genai`, `sec`, `rl`, `frontier`.

**Path rule (important):** local `solve` runs the copied `solve.py` with
`cwd = lvl_XX/`. Everything after `--` is relative to the challenge directory:
use `data/train.csv`, not `lvl_XX/data/train.csv`. `check --sample`, `fast_eda.py`,
`ctf_tactics.py` and `lmml_rs` run from the repo root and do use `lvl_XX/data/...`.
A wrong challenge-relative path often fails silently (e.g. a forensics scan of a
non-existent directory reports `files_scanned: 0`), so check `RESULT.json`.

### 4.3 Baseline / solve

`lmml_master.py solve` runs the copied `solve.py` with cwd = challenge directory,
then records `RESULT.json` and `submissions/submission.csv` in the vault.

```bash
# Tabular: fast CPU baseline
$PY lmml_master.py solve lvl_05 -- \
  --train data/train.csv --test data/test.csv --sample-sub data/sample_submission.csv \
  --target target --id-col id --models lgb --folds 3 --max-trees 400 --no-use-gpu

# NLP: TF-IDF + linear + hashed n-grams (no LGB for speed)
$PY lmml_master.py solve lvl_06 -- \
  --mode tfidf --text-col text --target-col target --id-col id \
  --folds 5 --svd-dim 32 --no-use-lgb --no-pretrained

# CV: scratch small CNN, low resolution, 2 folds
$PY lmml_master.py solve lvl_07 -- \
  --mode train --data-dir data --img-size 96 --epochs 2 --folds 2 \
  --backbone resnet18 --batch-size 16 --num-workers 0 --no-pretrained

# Audio: scratch conformer, short budget
$PY lmml_master.py solve lvl_08 -- \
  --mode audio_clf --data-dir data --arch conformer --epochs 4 --folds 2 \
  --batch-size 16 --no-pretrained

# Sec: forensics / prompt audit (no model needed)
$PY lmml_master.py solve lvl_09 -- --mode forensics --input data
$PY lmml_master.py solve lvl_09 -- --mode prompt_audit --input data/prompts.csv

# RL: short PPO run on the provided env
$PY lmml_master.py solve lvl_10 -- \
  --env CartPole-v1 --algo ppo --steps 20000 --n-envs 4 --seeds 42 --device cpu

# Frontier: one mode at a time
$PY lmml_master.py solve lvl_11 -- --mode pinn --pinn-pde poisson1d --adam-steps 200 --lbfgs-steps 25 --device cpu
$PY lmml_master.py solve lvl_11 -- --mode qubo --qubo-matrix data/Q.npy --device cpu
```

Notes:
- `--no-use-gpu` is the argparse negation of `--use-gpu` in the tabular template.
  On a CPU-only box the template falls back automatically, but passing it avoids
  CUDA probing and noisy warnings.
- For CV, `--num-workers 0` avoids 4 worker processes fighting 4 cores. If I/O is
  the bottleneck, use `--num-workers 2`.
- The audio template always computes the fast spectral tabular probe; the neural
  model is additional. For a pure probe, call `extract_audio_tabular_features()`
  from the template in a short script (see `reports/runbooks/04_audio_genai.md`).
- ASR/Whisper is installed but Fair-Play-gated: pass `--pretrained` (only if
  allowed) or `--whisper-model-dir /path/to/ct2-model`, otherwise
  `transcribe_asr()` raises `PermissionError`.
- `05_adversarial_sec.py` defaults to `--mode all` and runs the demo when neither
  `--input` nor `--prompt` is given; always pass `--input` (and `--model-path` for
  `evasion`/`backdoor_detect`/`privacy`).
- In `07_frontier_gnn_pinn.py`, `--mode all` is the demo suite, and `gnn`/`qml`/
  `qubo` without their data arguments fall back to synthetic benchmarks. Always
  pass `--nodes/--edges`, `--train/--test`, or `--qubo-matrix` for real work.
- In `03_cv_fast.py`, `--mode counting` (blob/peak object counting) is a new
  CPU-friendly mode; `--mode extract_probe` still produces noise without
  `--pretrained`.
- Do not pass `--pretrained` or `--allow-external-api` unless `fairplay.md` says so.

### 4.4 Validate before every upload

```bash
$PY lmml_master.py check lvl_05 --sample lvl_05/data/sample_submission.csv
./rust_core/target/release/lmml_rs validate lvl_05/submissions/submission.csv \
  lvl_05/data/sample_submission.csv
```

`check` fails on null/NaN/Inf, wrong row count, wrong column names, and constant
predictions. Constant predictions are sometimes a legitimate fallback but need
`--allow-constant` and a note in the report.

### 4.5 Submit to the competition API (not the git repo)

```bash
# CSV/file tasks
$PY tools/sigmoid_api.py submit-file --task-id task_xx --file lvl_05/submissions/submission.csv

# Flag/text tasks
$PY tools/sigmoid_api.py submit-text --task-id task_xx --text 'LMML{...}'
```

`lmml_master.py submit` / `git_sync.py` update the **team repo and vault**, not the
Sigmoid portal. The orchestrator owns commits/pushes; subagents only write files.

### 4.6 Vault discipline

```bash
$PY lmml_master.py vault lvl_05
```

Keep `submissions/submission.csv` as the current best, and copy milestones to
`.vault/best_submission_<tag>.csv`. Record CV, LB, command, and file SHA-256 in
`reports/levels/lvl_05.md`.

---

## 5. `optimizations/` — what replaces a heavy step

| Heavy step | Use this instead | Verified behaviour on this box |
| :--- | :--- | :--- |
| `HashingVectorizer` / Python n-gram loops | `optimizations/rust_fast_ext.fast_hash_ngrams_batch(texts, min_n, max_n, dim)` | 20k docs → CSR `(20000, 131072)` in **2.2 s** (loaded box) |
| pandas groupby target encoding | `fast_oof_target_encode(cats, y, folds, smooth=10)` or template 02 | 50k rows in **0.014 s**, leak-free per fold |
| Grid-search blend weights | `optimizations/blender.auto_blend(oof_list, y, metric="auc")` | Benchmarks Caruana, Nelder-Mead prob/rank/log-odds/power, non-neg stacker; nested OOF |
| Manual threshold grid | `optimizations/math_tricks.oof_optimal_threshold(y, prob)` | Exact `O(N log N)` nested threshold; returns a dict |
| Probability calibration | `math_tricks.calibrate_oof_and_test(oof, y, test, method="platt")` | Leak-free K-fold calibration (platt/isotonic/temperature) |
| Byte-entropy / stego / pickle scan | `ctf_tactics.py` + `lmml_rs ctf-scan <dir> --json` | `lmml_rs`: 40 files in **0.22 s**; `ctf_tactics.py --self-test`: passes |
| pandas-profiling EDA | `fast_eda.py` + `lmml_rs eda` | schema, nulls, leakage, drift, adversarial AUC in seconds |
| RankGauss / rolling stats / GAE / Dijkstra / BFS | `rust_fast_ext.fast_rank_gauss`, `fast_rolling_stats`, `fast_gae_returns`, `fast_dijkstra_sssp`, `fast_bfs_components` | Native Rust with NumPy fallback |
| Submission validation | `lmml_rs validate` (already called by `check`) | zero-NaN/Inf/schema check |
| Custom tight loops (DP, search, simulation) | `optimizations/cpp_fast_ext.py` (`g++` exists) | JIT C++20, ~1.7 s first compile, cached |

Caveats:
- Rust is a win for large arrays/text; for <10k rows, Python/NumPy is often faster
  than FFI overhead. Measure once.
- `auto_blend` needs the metric (`metric="auc"`, `"logloss"`, `"rmse"`) or a
  `metric_fn`; it raises if neither is provided.
- If the Rust library is unavailable, `LMML_NO_RUST=1` forces the NumPy fallbacks.
  Do **not** try to rebuild Rust during the marathon (`rustc` is not on PATH; the
  release `.so` and `lmml_rs` already exist).
- `ctf_tactics.py` can false-positive on normal text containing words like
  "subprocess"; triage CRITICAL findings, do not blindly delete files.

Smoke tests run on 2026-11-14 from `/home/kozak/mllm/MML`: `fast_hash_ngrams_batch`
(20k docs), `fast_oof_target_encode` (50k rows), `fast_hill_climb_blend`,
`blender.auto_blend` (AUC and logloss), `math_tricks.calibrate_oof_and_test`,
`ctf_tactics.py --self-test` (ALL PASSED), and `lmml_rs ctf-scan` (40 files, 0.22 s).
`fast_oof_target_encode` returns one OOF column, not a `(oof, test)` tuple.

---

## 6. Resource discipline (the real bottleneck)

```bash
uptime                    # load average; >4 means the box is saturated
free -g                   # keep ≥2 GB free; LightGBM on 1M×100 needs ~4–6 GB
ps -eo pcpu,pmem,comm --sort=-pcpu | head
```

- Set threads explicitly in every script:
  `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4`,
  `POLARS_MAX_THREADS=4`, `torch.set_num_threads(4)`.
- LightGBM: `num_threads=4`, `force_col_wise=True`. Never `n_jobs=-1`.
- Do not run CatBoost and LightGBM simultaneously; each will claim 4 threads.
- For a long job, `nice -n 10` it so interactive `check`/API calls stay responsive.
- Convert big CSVs to Parquet once; use Polars `scan_parquet` + `float32/int32`.
- Keep caches in `/home/kozak/.cache` (HF) or `work/cache`; disk has ~78 GB free.

---

## 7. Failure fallbacks

| Failure | Fallback |
| :--- | :--- |
| Template crashes on unknown schema | Write a 40-line custom `solve.py`; keep `RESULT.json` + `submissions/submission.csv` contract |
| No `torch_geometric` / `pennylane` / `faster_whisper` / `sb3_contrib` / `skimage` / `ortools` | Only `ortools` is missing now; `torch_geometric`, `pennylane`, `faster_whisper`, `sb3_contrib` and `skimage` are installed. Use self-contained template implementations or PyTorch/NumPy/scipy equivalents; do not `pip install` mid-task unless necessary |
| Dataset too large for RAM | Random/deterministic sample for CV; train on sampled rows; predict in chunks; Polars lazy scan |
| Metric unknown | Optimize a sensible proxy (AUC for binary, logloss for multiclass, RMSE for regression); state assumption |
| No signal / CV ≈ random | Submit prior/constant (with `--allow-constant`), move on; revisit only if time remains |
| CV great, LB terrible | Check ID leakage, duplicate rows, train/test overlap, target inversion, wrong submission order |
| Template demo fails | Demos are synthetic self-tests. Known: `05_adversarial_sec.py --demo` STRIP assertion is flaky at the default seed 42 (0.53 vs 0.85); `--seed 1` passes (0.9013). Real modes still work; do not burn time on the demo |
| Time nearly up | Freeze the best valid file, run `check`, submit, write report |

---

## 8. Pre-submit checklist (all must pass)

1. File exists, non-empty, exact row count and column order of `sample_submission`.
2. Zero null/NaN/Inf; probabilities in `[0,1]`; no constant column.
3. Predictions are in the same row order as the sample/test IDs.
4. Metric direction is correct (minimize vs maximize); no `1-p` inversion.
5. CV scheme matches the test distribution; adversarial AUC < 0.7 or explained.
6. OOF score is plausible (not 1.0000 from leakage).
7. `lmml_master.py check` and `lmml_rs validate` pass.
8. File SHA-256 and command are written in the task report; previous best is backed up.

---

## 9. What NOT to attempt without a GPU

- Fine-tuning DeBERTa/ModernBERT/RoBERTa (>1 epoch, >5k samples), or training a
  transformer from scratch. Frozen embeddings + linear probe only if pretrained is
  allowed.
- End-to-end ViT/ConvNeXt/EfficientNet at 224–384 px, 5-fold, >2k images.
- YOLO training, SAHI tiling at scale, video models, segmentation at 512 px.
- Whisper (any size) transcription of hours of audio; wav2vec/HuBERT fine-tuning.
- Diffusion/image generation, LLM QLoRA, vLLM, anything with "7B" in the name.
- PPO on pixel observations, Atari, MuJoCo, or >1M environment steps.
- GNNs >4 layers / >10k graphs, float64 PINNs with >50k collocation points,
  statevector QML >16 qubits.
- Huge Optuna sweeps (50+ trials × 5 folds) or 10-fold × 5-seed grids. Use 3-fold,
  one seed, fold-0 pruning, and a fixed budget.

When the task is GPU-shaped: ship the strongest classical baseline, make the
submission format perfect, and spend the remaining time on a track where CPU can
actually move the score.

---

## 10. Status and next actions

- **Done (2026-11-14):** read `AGENTS.md`, `OPTIMIZATION_PLAYBOOK.md`, all 10
  `pro_playbooks/`, and the 7 `templates/`; smoke-tested the utilities listed in
  §5 on this box; measured the CPU shapes quoted above.
- **Blockers:** the Sigmoid journey is not open yet (`/task` returns 403), so no
  real task data/metrics are available. `ortools` is the only missing optional
  package; `torch_geometric`, `pennylane`, `faster_whisper`, `sb3_contrib` and
  `skimage` are installed after the upstream dependency sync. The templates are
  self-contained, so optional packages are not required.
- **Environment caveat:** 4 cores are shared with other agents. Benchmark with
  `lmml_master.py bench` before trusting any estimate, and serialize heavy jobs.
- **On task unlock:** run intake §1.1–1.5, launch the matching baseline from §4.3,
  submit a valid file within 20 min, then follow the 120-minute workflow.
- **Next reports:** write `reports/levels/lvl_XX.md` per task with the exact
  command, CV/LB, runtime, fair-play decision, and file SHA-256.
