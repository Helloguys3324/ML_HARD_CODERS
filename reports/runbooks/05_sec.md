# Runbook 05 — Security / Adversarial / CTF (South America)

> Template: `templates/05_adversarial_sec.py` · Track key: `sec`
> CPU verdict: **excellent for forensics, prompt audit, poison filtering and small
> gradient attacks; not for large-scale evasion or LLM red-teaming.**

---

## 1. Triage (first 5 min)

```bash
export PY=/home/kozak/mllm/py REPO=/home/kozak/mllm/MML
cd "$REPO"
$PY optimizations/ctf_tactics.py --data lvl_09/data          # flags, stego, pickle, prompt injection
./rust_core/target/release/lmml_rs ctf-scan lvl_09/data --json
$PY lmml_master.py solve lvl_09 -- --mode forensics --input data
```

**Path rule:** `lmml_master.py solve` runs the copied `solve.py` with
`cwd = lvl_09/`, so everything after the `--` is relative to the challenge
directory (`data`, not `lvl_09/data`). The scanner commands (`fast_eda.py`,
`ctf_tactics.py`, `lmml_rs`, `check --sample`) run from the repo root and do use
`lvl_09/data`. Passing `lvl_09/data` to the template does **not** error — it
scans zero files and writes an empty result — so verify the `files_scanned`
count in `RESULT.json`.

Read the task statement for the required artifact:
- **Flag/text** → submit the exact string with `tools/sigmoid_api.py submit-text`.
- **CSV** (poison mask, prompt risk, adversarial predictions) → file submission.
- **Adversarial examples** (`.npy`/images) → check the expected array shape and
  value range (`[0,1]` vs `[0,255]`); the template saves `adv_samples.npy`.

Modes:
| Mode | What it does | CPU fit |
| :--- | :--- | :--- |
| `forensics` | file/container/stego/pickle/PDF/ZIP scan | excellent, seconds |
| `prompt_audit` | unicode/base64/injection/canary scan | excellent, seconds |
| `poison_filter` | spectral signatures / activation clustering on tabular/features | good |
| `backdoor_detect` | Neural Cleanse + STRIP on a provided model | good for small models |
| `evasion` | FGSM/PGD/MI-DI-TI/C&W/Square | only small batches |
| `privacy` | membership-inference + model-extraction probes on a provided model | small models |
| `all` | **offline demo suite**, not a real run | avoid for submissions |

`--mode` defaults to `all`. With no `--input`/`--prompt` the template silently
runs the demo suite and writes a synthetic submission; always pass `--input`
(and `--model-path` for `evasion`/`backdoor_detect`/`privacy`) for real work.

---

## 2. Fastest credible baseline (1–5 min)

For most security tasks the baseline is a **forensic scan**, not a model:

```bash
# Full scanner on the provided artifacts (repo root)
$PY optimizations/ctf_tactics.py --data lvl_09/data --json > work/sec_scan.json

# Prompt audit on a CSV/text file (challenge-relative paths after `--`)
$PY lmml_master.py solve lvl_09 -- --mode prompt_audit --input data/prompts.csv

# Poison filter on tabular features
$PY lmml_master.py solve lvl_09 -- --mode poison_filter --input data/train.csv

# Forensics-only report
$PY lmml_master.py solve lvl_09 -- --mode forensics --input data

# Validate the generated file
$PY lmml_master.py check lvl_09 --sample lvl_09/data/sample_submission.csv
```

`ctf_tactics.py` covers magic-byte mismatch, ZIP/PDF trailing payloads, pickle RCE
opcodes, safetensors headers, EXIF/tEXt chunks, PNG LSB stego, WAV RIFF/LSB,
ultrasonic energy, base64/Unicode-hidden flags, and prompt injection. It runs in
seconds and is the highest-ROI step on this track.

---

## 3. Best CPU model / attack

1. **Rule-based forensics + `lmml_rs ctf-scan`**. No training, no GPU, highest hit
   rate on CTF-style tasks.
2. **Small surrogate + MI-DI-TI-FGSM** for evasion. Use `resnet18`/a tiny CNN at
   64–96 px, 100–300 samples, 10–20 steps. Measure ASR on a held-out split of the
   surrogate; transfer to a black-box target is not guaranteed.
3. **Poison filtering**: spectral signatures, activation clustering, isolated
   forest / Mahalanobis on features. Template `sanitize_poisoned_dataset`.
4. **Backdoor detection**: Neural Cleanse (class-wise trigger inversion) + STRIP
   entropy. Works on small models/classes; CPU cost grows with classes × steps.
5. **Prompt audit**: normalization (NFKC, strip zero-width/bidi/tag chars), regex
   injection patterns, base64 decode, canary search. Pure Python, milliseconds.

Template evasion command:

```bash
$PY lmml_master.py solve lvl_09 -- \
  --mode evasion --attack mi_di_ti --eps 0.03137 --steps 20 --restarts 1 \
  --input data/x.npz --model-path data/model.pt --batch-size 32
```

The model must be a TorchScript module or `weights_only` state dict; the template
refuses unsafe pickles before loading. If you only have a black-box API, do not
fake a white-box attack; report the query budget and use transfer/query attacks.

---

## 4. Feature engineering / signals that pay

- Sliding-window Shannon entropy (256/512 bytes): high entropy at EOF = encrypted
  or compressed payload; low entropy tail = appended text/flag.
- Magic bytes vs extension mismatch; polyglot files (PNG header + ZIP/PDF payload).
- Strings: `LMML{`, `flag{`, `CTF{`, base64 blobs, long hex, URLs, canary tokens.
- Container structure: ZIP comments, ZipSlip paths, compression ratio, nested
  archives; PDF `/JavaScript`, `/OpenAction`, `/EmbeddedFile`, trailing bytes.
- Pickle opcodes: `GLOBAL`, `REDUCE`, `BUILD`, `os.system`, `subprocess`, `eval`.
- Safetensors: header bounds, `__metadata__`, unreferenced gaps, trailing bytes.
- Images: EXIF/tEXt/zTXt/iTXt, LSB bit-planes, palette anomalies, EOF trailing.
- Audio: RIFF chunk sizes, PCM LSB, DTMF (Goertzel), Morse envelope, ultrasonic
  energy, reversed-audio asymmetry.
- Text/prompts: zero-width/bidi/tag Unicode, homoglyph mixed scripts, base64,
  "ignore previous instructions", system-prompt extraction, canary leakage.

---

## 5. Validation

- **Evasion**: ASR on a held-out sample of the same surrogate; report perturbation
  norm (`L∞`/`L2`), query count, and whether the target is white-box or black-box.
  A 99% white-box ASR can be 10–30% black-box transfer.
- **Backdoor/poison**: AUC/F1 against known poisoned rows if labels are provided;
  otherwise report anomaly scores and flagged counts. Neural Cleanse's MAD index
  needs ≥3 classes to be meaningful.
- **Prompt audit**: macro-F1 on labeled prompts; inspect false positives manually.
  Injection regexes are high-precision but not exhaustive.
- **Forensics**: every CRITICAL finding must be manually confirmed (decode the
  payload/flag). The scanner can false-positive on normal text containing
  "subprocess" or high-entropy JPEGs.
- **Submission**: if the task expects a mask/CSV, align IDs exactly; if it expects
  adversarial samples, verify shape/dtype/range and that the model still predicts
  the target class.

---

## 6. Common pitfalls

| Pitfall | Symptom | Fix |
| :--- | :--- | :--- |
| `--demo` STRIP assertion | `AssertionError: Expected STRIP AUC >= 0.85` at seed 42 | Known upstream flake (observed 0.53–0.55). `--seed 1` passed (0.9013 in `reports/02_ENVIRONMENT.md`). Real modes work; don't debug the demo |
| Unsafe `torch.load` | RCE risk | Use TorchScript/`weights_only=True`; template blocks unsafe pickles |
| Blindly deleting "suspicious" files | Lost signal | Confirm findings; entropy alone is not malicious |
| Huge PGD batch | Hours on CPU | 100–300 samples, 10–20 steps, 64–96 px |
| C&W/Square at scale | Very slow | Use FGSM/MI-FGSM first; C&W only on a few samples |
| Wrong perturbation range | ASR 0 or broken images | Match model normalization; clamp to valid range |
| LLM API used without permission | Disqualification | Omit `--allow-external-api` (template default is off) |
| Flag format assumed | Submission rejected | Read the exact regex/prefix from the task |

---

## 7. Snippets

### 7.1 One-shot forensic scan

```bash
mkdir -p work
$PY optimizations/ctf_tactics.py --data lvl_09/data --json > work/ctf.json
./rust_core/target/release/lmml_rs ctf-scan lvl_09/data --json > work/ctf_rust.json
# Triage CRITICAL findings first; decode base64/hex; grep for LMML{...}
```

### 7.2 Prompt audit (custom, no model)

```python
import re, unicodedata, base64
ZW = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)
def audit(text):
    t = unicodedata.normalize("NFKC", text).translate(ZW)
    flags = []
    if re.search(r"ignore (all )?previous|system prompt|developer mode", t, re.I):
        flags.append("injection")
    if re.search(r"[A-Za-z0-9+/]{40,}={0,2}", t):
        try: base64.b64decode(re.search(r"[A-Za-z0-9+/]{40,}={0,2}", t).group())
        except Exception: pass
        flags.append("base64_blob")
    return {"text": t, "flags": flags, "risk": len(flags)}
```

### 7.3 Small PGD sanity check before a full run

```bash
# 20 samples, 5 steps: verifies the model loads, gradients flow, and files save
$PY lmml_master.py solve lvl_09 -- --mode evasion --attack pgd --steps 5 \
  --input data/x_small.npz --model-path data/model.pt --batch-size 8
```

---

## 8. Runtime on 4 cores (shared box)

| Operation | Time |
| :--- | :--- |
| `ctf_tactics.py` on 40 files | seconds |
| `lmml_rs ctf-scan` on 40 files | ~0.2 s |
| Prompt audit, 10k prompts | seconds |
| Poison filter, 10k×50 | <1 min |
| Neural Cleanse, 4 classes × 100 steps | 1–5 min |
| MI-FGSM, 100 images × 20 steps @ 64 px | 3–10 min |
| MI-FGSM, 1k images × 20 steps @ 96 px | 30–90 min |
| C&W, 100 images × 50 steps | 20–60 min |

## 9. Do not attempt without GPU

- PGD/C&W/Square on >1k images at 224 px, or ensemble attacks over 3+ large models.
- Neural Cleanse/STRIP on large transformers or many classes.
- LLM jailbreak sweeps, prompt-injection fuzzing at scale, or external API probing
  without explicit permission.
- Training a robust model or an adversarial detector from scratch on images.
