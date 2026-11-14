# 16 — Modal Cloud GPU Setup

_Orchestrator, 2026-11-14 ~09:47 UTC._

## Status

- **Profile:** `democracy1984` (active).
- **Token:** verified against `api.modal.com`; token ID starts with `ak-…`,
  secret supplied by the user. Stored in `~/.modal.toml` (outside the repo).
  **Never** copy the secret into a report, commit, or chat.
- **Volumes:** `lmml-data`, `lmml-cache` (created 2026-11-13, visible).
- **Client:** `modal 1.6.4` installed in `/home/kozak/mllm/venv`.
- **Readiness check:** `modal run modal_runners/modal_train.py --smoke-test --gpu T4`
  (started 09:47 UTC; result appended below). This builds the image, starts a
  T4, runs `nvidia-smi`, imports torch, and reports the device.

## Setup command (secret redacted)

```bash
cd /home/kozak/mllm
./py -m modal token set \
  --token-id ak-… \
  --token-secret as-… \
  --profile=democracy1984 --activate --verify
```

Expected output:

```text
Token verified successfully!
Token written to /home/kozak/.modal.toml in profile democracy1984.
```

## Useful commands

```bash
cd /home/kozak/mllm/MML

# Confirm profile / volumes
../py -m modal profile current
../py -m modal volume list

# Generic GPU smoke test (T4)
../py -m modal run modal_runners/modal_train.py --smoke-test --gpu T4

# Run a challenge solver on a cloud GPU (syncs data + Rust build)
../py -m modal run modal_runners/modal_train.py \
  --script work/tasks/<task_id>/solve.py --challenge-dir work/tasks/<task_id> \
  --gpu A10G --sync-data --build-rust --args="--epochs 5"

# CV training / parallel folds
../py -m modal run modal_runners/modal_cv.py \
  --mode timm --data-dir /data/images --model convnext_small.fb_in22k_ft_in1k \
  --epochs 10 --gpu A10G --pretrained   # ONLY if the task allows pretrained

# LLM QLoRA / inference (requires --pretrained-ok and fair-play clearance)
../py -m modal run modal_runners/modal_llm.py --mode infer \
  --model Qwen/Qwen2.5-7B-Instruct --data-path /data/test.jsonl \
  --out-file /data/outputs/preds.jsonl --gpu A10G --pretrained-ok
```

## Cost/selection notes

- Prefer **T4/L4** for smoke tests and small CV; **A10G/L40S** for real CV/LLM
  training; **A100/H100** only when the time budget justifies it.
- Volumes are persistent: put datasets under `/data` (`lmml-data`) and caches
  under `/cache` (`lmml-cache`), not in the image.
- Modal runners are Fair-Play-gated: `--pretrained` / `--pretrained-ok` must not
  be passed unless the task statement explicitly allows pretrained models.
- Keep the local watcher running; cloud use does not replace the intake/claim
  protocol. Submit only through `tools/submit_officer.py`.

## Smoke test result (2026-11-14 ~09:50 UTC)

`modal run modal_runners/modal_train.py --smoke-test --gpu T4` → **PASS**:

```text
nvidia_smi        : Tesla T4, 15360 MiB, driver 620.70.06
cuda_available    : True
device_name       : Tesla T4
compute_capability: 7.5
amp_dtype         : torch.float16 (bf16_native=False)
matmul_4k_tflops  : 20.98
torch_version     : 2.15.0+cu132
rustc             : 1.100.0
g++               : 12.2.0
volumes_mounted   : {'/data': True, '/cache': True}
elapsed_s         : 1.28
```

The persistent volumes, Rust build cache, and GPU image are working. This is the
green light for cloud GPU escalation when a task needs it (CV training, LLM
fine-tuning, large adversarial/GNN sweeps).
