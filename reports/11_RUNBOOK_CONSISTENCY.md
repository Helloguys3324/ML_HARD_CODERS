# Runbook Consistency Audit — LMML 2026 (post-upstream `4e0946f`)

> Auditor: subagent J · Date: 2026-11-14 · Scope: CPU playbook + 7 track runbooks +
> task-brief/checklist templates.
> Templates and code were **not modified**. No commits, no pushes, no tokens touched.
> All Python run through `/home/kozak/mllm/py`.

## 0. TL;DR

- All seven templates were re-read from `--help` and from their `build_parser()`.
  **Every mode name used in the docs is still valid.**
- **All 29 pre-existing template invocations parse cleanly** (argparse), and so do
  the 10 commands added to the cheat sheet by this audit (39/39). No invented
  flags remain.
- **One class of genuinely broken command was found:** `runbooks/05_sec.md` passed
  challenge-relative paths (`lvl_09/data`) to `lmml_master.py solve`, which runs
  `solve.py` with `cwd = lvl_09/`. The result does not crash — it silently scans
  **zero files** and writes an empty submission. Fixed to `data/...`.
- **Stale environment claims** (packages "not installed") were corrected:
  `torch_geometric`, `pennylane`, `faster_whisper`, `sb3_contrib`, `skimage` are
  installed; only `ortools` is missing.
- **Two Fair-Play behavior changes** after the rewrite were documented:
  `04 --arch ssl_probe` now requires `--pretrained`, and ASR requires
  `--pretrained` or `--whisper-model-dir`.
- `SUBMISSION_CHECKLIST.md` was audited: it contains no template commands and its
  `lmml_master.py check` / `validate_submission.py` / `sigmoid_api.py` flags are all
  current. **No changes.**

## 1. Method and evidence

Commands run from `/home/kozak/mllm/MML`:

```bash
for f in templates/0*.py; do /home/kozak/mllm/py "$f" --help; done   # all 7, captured
/home/kozak/mllm/py /tmp/opencode/check_cmds.py                       # argparse-only replay of 39 doc commands
/home/kozak/mllm/py lmml_master.py check --help
/home/kozak/mllm/py lmml_master.py new --help
/home/kozak/mllm/py lmml_master.py tracks
/home/kozak/mllm/py tools/sigmoid_api.py --help
/home/kozak/mllm/py tools/sigmoid_api.py task-overview --help
./rust_core/target/release/lmml_rs validate|eda|ctf-scan --help
/home/kozak/mllm/py optimizations/fast_eda.py --help
/home/kozak/mllm/py optimizations/ctf_tactics.py --help
/home/kozak/mllm/py tools/validate_submission.py --help
/home/kozak/mllm/py - <<'PY'   # helper signatures via inspect.signature
... fast_hash_ngrams_batch, fast_oof_target_encode, auto_blend,
    oof_optimal_threshold, calibrate_oof_and_test ...
PY
```

Behavioral verification (short, synthetic, nothing long or GPU-bound):

| Check | Command | Result |
| :--- | :--- | :--- |
| CV demo + counting | `03_cv_fast.py --demo --img-size 32` / `--mode counting --demo` | pass, 9.7 s / 0.2 s |
| Tabular demo | `02_tabular_fast.py --demo --demo-task binary` | pass, AUC 0.9307, 12.1 s |
| Sec forensics | `05_adversarial_sec.py --mode forensics --input <tmp dir>` | pass, found planted `LMML{...}`, 1 CRITICAL |
| Frontier PINN | `07_frontier_gnn_pinn.py --mode pinn --adam-steps 5 --lbfgs-steps 0 --collocation 32` | pass, RMSE 0.551, 1.5 s |
| Audio ASR gate | `04 --mode asr --audio <tmp wav>` without gate flags | expected `PermissionError` (gate works) |
| ctf_tactics self-test | `ctf_tactics.py --self-test` | `ALL SELF-TESTS PASSED` |
| E2E tabular via solve | `lmml_master.py solve <tmp> -- --train data/train.csv ... --models lgb` | AUC 0.896; `check` PASS |
| E2E NLP via solve | `lmml_master.py solve <tmp> -- --mode tfidf ... --no-use-lgb` | AUC 0.884; `RESULT.json` written |
| E2E sec via solve | `lmml_master.py solve <tmp> -- --mode forensics --input data` | scanned 1 file, `RESULT.json` `files_scanned=1` |
| Wrong-path control | same with `--input lvl_99/data` | exit 0 but `files_scanned=0` (silent empty result) |

The wrong-path control is the hard evidence for the runbook 05 fix: the template
does not fail on a nonexistent directory, it reports zero findings.

## 2. Real template surface (source of truth)

### 2.1 Modes and Fair-Play gates

| Template | `--mode` choices (default) | Gates (default off) | Default data paths |
| :--- | :--- | :--- | :--- |
| `01_nlp_fast.py` | `tfidf` (**default**), `transformer`, `hybrid`, `retrieval`, `ner` | `--pretrained` (off) | `data/train.csv`, `data/test.csv` |
| `02_tabular_fast.py` | `tabular` (**default**), `unsupervised` | none | `data/train.csv`, `data/test.csv` |
| `03_cv_fast.py` | `train` (**default**), `extract_probe`, `yolo`, `counting` | `--pretrained` (off) | none (`--data-dir/--train-csv/--img-dir`) |
| `04_audio_genai_fast.py` | `audio_clf`, `asr`, `audio_forensics`, `genai_rag`, `all` (**demo**); auto: `audio_clf`, or `asr` if `--audio`, or `all` if `--demo` | `--pretrained`, `--allow-external-api` (off) | none (`--data-dir/--audio`) |
| `05_adversarial_sec.py` | `all` (**default = demo**), `evasion`, `backdoor_detect`, `poison_filter`, `forensics`, `prompt_audit`, `privacy` | `--pretrained` (off), `--allow-external-api` (`store_true`) | none (`--input`, `--model-path`) |
| `06_rl_fast.py` | n/a; `--algo` choices `ppo` (**default**), `dqn`, `a2c`, `reinforce`, `value_iter`, `policy_iter`, `qlearning`, `sarsa`, `bandit_eps`, `bandit_ucb`, `astar` | none | `--env` auto-selected by algo |
| `07_frontier_gnn_pinn.py` | `all` (**default = demo**), `gnn`, `pinn`, `qml`, `qubo` | none | none (`--nodes/--edges/--train/--test/--qubo-matrix`) |

Key defaults verified from argparse: 01 `folds=5, svd-dim=64, hash-dim=131072,
epochs=3`; 02 `folds=5, models=lgb,xgb,cat,ridge,mlp, cv=stratified, use-gpu=True,
calibrate=auto, tune-threshold=True`; 03 `folds=5, epochs=5, tta=1, num-workers=-1,
img-size=0(auto)`; 04 `sr=16000, duration-s=3.0, n-mels=64, epochs=8, folds=3,
arch=conformer, whisper-size=base`; 05 `eps=8/255, steps=20, attack=mifgsm`; 06
`steps=6000, n-envs=4, seeds=42, n-steps=128, batch-size=128`; 07 `gnn-epochs=25,
gnn-folds=3, pinn-pde=poisson1d, dtype=float32, adam-steps=200, lbfgs-steps=25,
collocation=256, qml-qubits=4, qubo-vars=32`.

### 2.2 Path rule (the one that bites)

`lmml_master.py solve` runs the copied `solve.py` with `cwd = lvl_XX/`
(`lmml_master.py` line 552). Therefore:

- **After `--`:** use challenge-relative paths (`data/train.csv`, `data/Q.npy`).
  `lvl_XX/data/...` is wrong there and may fail **silently**.
- **Repo-root commands:** `lmml_master.py check --sample lvl_XX/data/...`,
  `fast_eda.py --data lvl_XX/data`, `ctf_tactics.py --data lvl_XX/data`,
  `lmml_rs ctf-scan lvl_XX/data`, `tools/sigmoid_api.py` all use repo-relative
  paths and are correct as written.
- Keep `--out-dir` at its default (challenge dir) when using `solve`; the template
  writes `RESULT.json` into `out-dir` and the master only reads
  `lvl_XX/RESULT.json`.

## 3. Table of changes

| # | File | Location | Old | New | Reason |
| ---: | :--- | :--- | :--- | :--- | :--- |
| 1 | `reports/runbooks/05_sec.md` | §1 triage | `--mode forensics --input lvl_09/data` | `--input data` | `solve` cwd is `lvl_09/`; old path scans 0 files silently |
| 2 | `reports/runbooks/05_sec.md` | §1 (new paragraph) | — | Path rule: challenge-relative after `--`, repo-relative for scanners/`check`; verify `files_scanned` | Prevents recurrence |
| 3 | `reports/runbooks/05_sec.md` | §2 prompt audit | `--input lvl_09/data/prompts.csv` | `--input data/prompts.csv` | Same cwd bug |
| 4 | `reports/runbooks/05_sec.md` | §2 poison filter | `--input lvl_09/data/train.csv` | `--input data/train.csv` | Same cwd bug |
| 5 | `reports/runbooks/05_sec.md` | §2 forensics | `--input lvl_09/data` | `--input data` | Same cwd bug |
| 6 | `reports/runbooks/05_sec.md` | §1 modes table | 5 rows | Added `privacy`, `all` (demo) | New modes in rewritten template |
| 7 | `reports/runbooks/05_sec.md` | §1 modes | — | Note: default `--mode all` + no `--input` silently runs demo | Surprising behavior; synthetic submission risk |
| 8 | `reports/runbooks/05_sec.md` | §6 pitfall | "flaky demo (0.53)" | "flake at default seed 42; `--seed 1` passes (0.9013)" | Actionable workaround from `02_ENVIRONMENT.md` |
| 9 | `reports/runbooks/04_audio_genai.md` | §1 ASR | "`faster-whisper` is **not installed**" | Installed; ASR gated by `--pretrained` or `--whisper-model-dir`, else `PermissionError` | Upstream dependency sync + new Fair-Play gate |
| 10 | `reports/runbooks/04_audio_genai.md` | §3 commands | — | Added verified ASR invocation with `--pretrained` and local-model alternative | Gate needs explicit flags |
| 11 | `reports/runbooks/04_audio_genai.md` | §3 note | "`--no-pretrained` does **not** gate `ssl_probe`" | `ssl_probe` now raises `PermissionError` without `--pretrained` | Template now checks `cfg.pretrained` (line 1160) |
| 12 | `reports/runbooks/04_audio_genai.md` | §6 pitfalls | ssl_probe = fair-play risk; ASR = ImportError | ssl_probe = `PermissionError`; ASR = gate error, package installed | Align with actual behavior |
| 13 | `reports/runbooks/04_audio_genai.md` | §3 | — | Added path rule (`data/...` vs `lvl_08/data/...`) | Consistency with §2.2 |
| 14 | `reports/runbooks/06_rl.md` | §3 masking | "`sb3_contrib` is **not installed**; use CleanRL" | Installed 2.9.2; `--action-masking --algo ppo` uses `MaskablePPO`; supports ppo/a2c/reinforce; DQN raises | Upstream sync + code at line 1612/2399 |
| 15 | `reports/runbooks/06_rl.md` | Header | — | Full algo list + built-in envs + `--env` auto-selection | New algos/envs since rewrite |
| 16 | `reports/runbooks/07_frontier.md` | Header | "`torch_geometric` and `pennylane` are **not installed**" | Installed, but template deliberately self-contained | Upstream sync |
| 17 | `reports/runbooks/07_frontier.md` | §1 | — | `--mode all` is the default demo suite; gnn/qml/qubo without data args run synthetic benchmarks | Prevents synthetic submissions |
| 18 | `reports/runbooks/07_frontier.md` | §6 pitfalls | ModuleNotFoundError rows | "only if absent; it is installed now" | Accuracy |
| 19 | `reports/03_PLAYBOOK_CPU.md` | §4.2 | — | Path rule paragraph | Cross-runbook correctness |
| 20 | `reports/03_PLAYBOOK_CPU.md` | §4.3 notes | — | ASR gate; sec demo default; frontier synthetic fallbacks; CV `counting` mode | New behavior after rewrite |
| 21 | `reports/03_PLAYBOOK_CPU.md` | §7 fallback | "No torch_geometric / pennylane / faster_whisper / sb3_contrib / skimage / ortools" | Only `ortools` missing; rest installed | Stale package inventory |
| 22 | `reports/03_PLAYBOOK_CPU.md` | §7 fallback | demo flake "0.53 vs 0.85" | Seed-42 flake; `--seed 1` passes (0.9013) | Actionable |
| 23 | `reports/03_PLAYBOOK_CPU.md` | §10 blockers | Packages "not installed" | Only `ortools` missing | Stale package inventory |
| 24 | `reports/runbooks/01_nlp.md` | Header | — | Mode list incl. `retrieval` (`--corpus`) and `ner` | New mode `ner` |
| 25 | `reports/runbooks/02_tabular.md` | Header | — | Modes `tabular`/`unsupervised`; `--use-gpu` default on | New mode + default clarity |
| 26 | `reports/runbooks/03_cv.md` | Header/§1/§3 | — | Mode list; counting bullet; verified `--mode counting` command | New mode `counting` |
| 27 | `reports/templates/TASK_BRIEF.md` | §6 | Direct `templates/...` call from repo root with challenge-relative paths and `--out-dir work/...` | `lmml_master.py solve` example + per-track flag table + out-dir warning | Old example would resolve `data/` at repo root and break the `RESULT.json` contract |
| 28 | `reports/templates/SUBMISSION_CHECKLIST.md` | — | — | No change | No template commands; all tool flags current |
| 29 | `reports/runbooks/05_sec.md` | §7.1 snippet | `--data data`, `ctf-scan data` mixed with `./rust_core/...` | `lvl_09/data` + `mkdir -p work` | Snippet was only runnable from two different cwds at once |

Also verified but left unchanged because they are correct:

- `--no-use-gpu`, `--no-use-lgb`, `--no-pretrained` are real
  `argparse.BooleanOptionalAction` negations, not invented flags.
- `lmml_master.py check --sub/--sample/--allow-constant`,
  `lmml_rs validate <sub> [sample]`, `tools/validate_submission.py
  --submission/--sample`, `tools/sigmoid_api.py submit-file/submit-text --task-id`,
  `fast_eda.py --data`, `ctf_tactics.py --data/--json`, `lmml_rs eda/ctf-scan`.
- Helper signatures: `fast_hash_ngrams_batch(texts, min_n, max_n, dim,
  analyzer="word_char")`, `fast_oof_target_encode(cats, y, folds, smooth=10.0)`,
  `auto_blend(oof, y, test, metric=..., task=...)`,
  `oof_optimal_threshold(y, p)["threshold_median"]`,
  `calibrate_oof_and_test(oof, y, test, method="platt")` — all match the docs.
- `05`'s "template refuses unsafe pickles" claim: `_load_real_model` runs
  `inspect_checkpoint_file` and uses `torch.load(..., weights_only=True)`.

## 4. Remaining uncertainties

1. **ASR success path not executed.** The gate failure was reproduced, but
   `--pretrained` would download Whisper weights and `--allow-external-api` needs
   provider keys, so neither was run. Command syntax is `--help`/argparse-verified.
2. **Pretrained CV / SSL probes not executed.** `--mode extract_probe --pretrained`
   and `--arch ssl_probe` require model downloads and fair-play permission; only
   argparse and source-level gating were checked.
3. **YOLO mode not executed.** `--yolo-weights yolo11n.pt` downloads weights.
4. **`05` evasion/backdoor/privacy and `07` gnn/qml/qubo real paths not executed.**
   No task checkpoints/graphs were available. Flag parsing and dispatch were read
   in source; `pinn`, `counting`, `forensics`, and all demos were executed.
5. **Demo timings** (81 s tabular all-task, 223 s RL, 225 s frontier) were not
   re-measured end-to-end; only the shorter demos above were timed. The numbers
   remain plausible but are not fresh measurements.
6. **`--mode all` semantics** for 04/05/07 are confirmed in source as demo suites;
   the exact runtime of each full suite was not measured.
7. **Defaults can drift again** if upstream rewrites the templates. The cheat sheet
   below is valid for the tree at commit `4e0946f` (templates dated 2026-11-14
   11:18).

## 5. Verified command cheat sheet

Run from `/home/kozak/mllm/MML` unless noted. `[help]` = argparse/`--help` verified;
`[run]` = executed on synthetic data; `[demo]` = demo executed.

```bash
export PY=/home/kozak/mllm/py
export REPO=/home/kozak/mllm/MML
cd "$REPO"

# health / scaffold
$PY lmml_master.py health                                  # [run] 10/10 packages
$PY lmml_master.py tracks                                  # [run]
$PY lmml_master.py new lvl_05 --track tabular              # [help] tracks: nlp tabular cv audio genai sec rl frontier
```

### Track 01 — NLP

```bash
# baseline (no LGB, no pretrained) — verified end-to-end on synthetic data [run]
$PY lmml_master.py solve lvl_06 -- \
  --mode tfidf --text-col text --target-col target --id-col id \
  --metric auto --folds 5 --svd-dim 32 --no-use-lgb --no-pretrained
# full TF-IDF + hash-LGB stack
$PY lmml_master.py solve lvl_06 -- --mode tfidf --text-col text --target-col target --folds 5
# retrieval (needs a corpus; test CSV holds queries)
$PY lmml_master.py solve lvl_06 -- --mode retrieval --corpus data/corpus.csv --text-col text --id-col id
# NER (CSV token/tag columns; sentence or CoNLL rows)
$PY lmml_master.py solve lvl_06 -- --mode ner --text-col tokens --target-col tags
```

### Track 02 — Tabular

```bash
# fast baseline — verified end-to-end + `check` PASS [run]
$PY lmml_master.py solve lvl_05 -- \
  --train data/train.csv --test data/test.csv --sample-sub data/sample_submission.csv \
  --target target --id-col id --task auto --metric auto --cv stratified \
  --models lgb --folds 3 --max-trees 400 --no-use-gpu
# full blend
$PY lmml_master.py solve lvl_05 -- \
  --train data/train.csv --test data/test.csv --sample-sub data/sample_submission.csv \
  --target target --id-col id --models lgb,xgb,cat,ridge \
  --folds 5 --max-trees 1200 --calibrate auto --tune-threshold --no-use-gpu
# unsupervised
$PY lmml_master.py solve lvl_05 -- --mode unsupervised --n-clusters 0 --contamination 0.05
```

### Track 03 — CV

```bash
# scratch CNN / tiny images
$PY lmml_master.py solve lvl_07 -- \
  --mode train --data-dir data --img-size 96 --epochs 2 --folds 2 \
  --backbone resnet18 --batch-size 16 --num-workers 0 --tta 2 --no-pretrained
# frozen embeddings (ONLY with fair-play permission)
$PY lmml_master.py solve lvl_07 -- \
  --mode extract_probe --data-dir data --pretrained \
  --backbone resnet18,convnext_tiny --img-size 160 --batch-size 32 \
  --num-workers 0 --probe-head all
# object counting [demo run]
$PY lmml_master.py solve lvl_07 -- \
  --mode counting --train-csv data/train.csv --img-dir data/images \
  --counting-method auto --counting-min-area 12
# demo [run]
$PY templates/03_cv_fast.py --demo --img-size 32 --out-dir /tmp/cv_demo
```

### Track 04 — Audio / GenAI

```bash
# scratch conformer
$PY lmml_master.py solve lvl_08 -- \
  --mode audio_clf --data-dir data --arch conformer \
  --sr 16000 --duration-s 3.0 --n-mels 64 --epochs 4 --folds 2 \
  --batch-size 16 --tta-crops 2 --no-pretrained
# forensics / RAG
$PY lmml_master.py solve lvl_08 -- --mode audio_forensics --data-dir data
$PY lmml_master.py solve lvl_08 -- --mode genai_rag --rag-corpus data/corpus.jsonl --rag-queries data/queries.csv
# ASR: explicit Fair-Play gate required (gate failure reproduced [run])
$PY lmml_master.py solve lvl_08 -- \
  --mode asr --audio data/clip.wav --whisper-size base --language en --pretrained
# local CT2 model instead of download:
#   --mode asr --audio data/clip.wav --whisper-model-dir /path/to/ct2-model
```

### Track 05 — Security / CTF

Scanners run from the repo root; template `--input` paths are challenge-relative.

```bash
$PY optimizations/ctf_tactics.py --data lvl_09/data --json > work/sec_scan.json
./rust_core/target/release/lmml_rs ctf-scan lvl_09/data --json

$PY lmml_master.py solve lvl_09 -- --mode forensics --input data          # [run] e2e
$PY lmml_master.py solve lvl_09 -- --mode prompt_audit --input data/prompts.csv
$PY lmml_master.py solve lvl_09 -- --mode poison_filter --input data/train.csv
$PY lmml_master.py solve lvl_09 -- --mode evasion --attack mi_di_ti \
  --eps 0.03137 --steps 20 --restarts 1 --input data/x.npz --model-path data/model.pt --batch-size 32
$PY lmml_master.py solve lvl_09 -- --mode evasion --attack pgd --steps 5 \
  --input data/x_small.npz --model-path data/model.pt --batch-size 8
$PY lmml_master.py solve lvl_09 -- --mode privacy --input data/mia.npz --model-path data/model.pt
```

### Track 06 — RL

```bash
$PY lmml_master.py solve lvl_10 -- \
  --env CartPole-v1 --algo ppo --backend sb3 --steps 20000 --n-envs 4 \
  --n-steps 128 --batch-size 128 --seeds 42 --device cpu --no-subproc
# built-ins / tabular and bandit algos (env auto-selected if --env omitted)
$PY lmml_master.py solve lvl_10 -- --algo qlearning --steps 20000
$PY lmml_master.py solve lvl_10 -- --algo astar --allow-diag
$PY lmml_master.py solve lvl_10 -- --env MaskedGridChoice-v0 --algo ppo --action-masking
```

### Track 07 — Frontier

```bash
$PY lmml_master.py solve lvl_11 -- \
  --mode gnn --nodes data/nodes.csv --edges data/edges.csv \
  --gnn-task node --gnn-arch gin --gnn-hidden 64 --gnn-layers 3 \
  --gnn-epochs 25 --gnn-folds 3 --device cpu
$PY lmml_master.py solve lvl_11 -- \
  --mode pinn --pinn-pde poisson1d --dtype float64 \
  --adam-steps 200 --lbfgs-steps 25 --collocation 256 --device cpu     # [run] 5-step variant
$PY lmml_master.py solve lvl_11 -- \
  --mode qubo --qubo-matrix data/Q.npy --qubo-vars 32 --device cpu
$PY lmml_master.py solve lvl_11 -- \
  --mode qml --train data/train.csv --test data/test.csv \
  --qml-qubits 6 --qml-layers 3 --device cpu
$PY lmml_master.py solve lvl_11 -- --demo        # ~225 s, optional
```

### Validate and submit

```bash
$PY lmml_master.py check lvl_XX --sample lvl_XX/data/sample_submission.csv
${REPO}/rust_core/target/release/lmml_rs validate \
  lvl_XX/submissions/submission.csv lvl_XX/data/sample_submission.csv

$PY tools/validate_submission.py \
  --submission lvl_XX/submissions/submission.csv \
  --sample lvl_XX/data/sample_submission.csv --json

$PY tools/sigmoid_api.py submit-file --task-id task_xx --file lvl_XX/submissions/submission.csv
$PY tools/sigmoid_api.py submit-text --task-id task_xx --text 'LMML{...}'
$PY lmml_master.py vault lvl_XX
```

### Gotchas worth repeating

- `solve` cwd = `lvl_XX/`; `check`/scanners cwd = repo root.
- `05` with no `--input`/`--prompt` runs the **demo** and writes a synthetic file.
- `07 --mode all` (default) runs the demo; gnn/qml/qubo without data args run
  synthetic benchmarks.
- `04 --mode all` is the offline smoke suite.
- `03 --mode extract_probe --no-pretrained` gives random-init "embeddings"; never
  submit them.
- `05 --demo` STRIP assertion flakes at seed 42; `--seed 1` passed upstream.
- Keep template `--out-dir` at the challenge dir when using `solve`, otherwise the
  master cannot read `RESULT.json`.
