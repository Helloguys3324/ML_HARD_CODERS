# Runbook 04 — Audio / GenAI (Africa)

> Template: `templates/04_audio_genai_fast.py` · Track keys: `audio`, `genai`
> CPU verdict: **log-mel statistics + GBDT is strong and fast; small scratch
> conformers are viable for ≤2k clips; ASR/Whisper and LLM work are not viable
> unless the task is tiny and dependencies/APIs are permitted.**
> Modes: `audio_clf` (default), `asr`, `audio_forensics`, `genai_rag`; `all` is
> the offline demo suite.

---

## 1. Triage (first 5 min)

```bash
find lvl_08/data -type f \( -iname '*.wav' -o -iname '*.mp3' -o -iname '*.flac' -o -iname '*.ogg' \) | wc -l
$PY optimizations/fast_eda.py --data lvl_08/data
$PY optimizations/ctf_tactics.py --data lvl_08/data
```

Sub-modality:
- **Audio classification** (scene, species, emotion, spoof): mel stats + GBDT,
  then small conformer.
- **ASR / transcription**: `faster-whisper` **is installed**, but the template now
  Fair-Play-gates ASR: pass `--pretrained` (only if rules allow pretrained ASR) or
  `--whisper-model-dir /path/to/local/ct2-model`; otherwise
  `transcribe_asr()` raises `PermissionError`. External Whisper APIs need
  `--allow-external-api` plus provider keys. CPU ASR is still slow; use only for
  short audio.
- **Audio forensics** (hidden flags, DTMF, Morse, ultrasonic, LSB): pure CPU and
  fast; template `--mode audio_forensics`.
- **GenAI/RAG** (retrieval, extraction, prompt audit): BM25 + SVD + RRF on CPU;
  external LLMs only if allowed.

---

## 2. Fastest credible baseline (2–5 min)

Log-mel / spectral statistics + LightGBM. This is the audio equivalent of TF-IDF:

```python
# work/audio_probe.py — run from the challenge directory
import numpy as np, polars as pl, librosa, lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def feats(p, sr=16000, dur=3.0):
    y, _ = librosa.load(p, sr=sr, mono=True, duration=dur)
    if len(y) < sr // 10: return np.zeros(64 * 5 * 3 + 9, np.float32)
    y = y / (np.sqrt(np.mean(y ** 2)) + 1e-8)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=512, hop_length=160)
    L = librosa.power_to_db(S, ref=np.max)
    d1 = librosa.feature.delta(L); d2 = librosa.feature.delta(L, order=2)
    out = []
    for M in (L, d1, d2):
        out += [M.mean(1), M.std(1), M.max(1), np.percentile(M, 10, axis=1),
                np.percentile(M, 90, axis=1)]
    out += [librosa.feature.zero_crossing_rate(y).mean(),
            librosa.feature.rms(y=y).mean(), librosa.feature.rms(y=y).std(),
            librosa.feature.spectral_centroid(y=y, sr=sr).mean(),
            librosa.feature.spectral_bandwidth(y=y, sr=sr).mean(),
            librosa.feature.spectral_rolloff(y=y, sr=sr).mean(),
            librosa.feature.spectral_flatness(y=y).mean(),
            float(np.mean(np.abs(y) < 1e-3)), len(y) / sr]
    return np.concatenate([np.ravel(v) for v in out]).astype(np.float32)

tr = pl.read_csv("data/train.csv"); te = pl.read_csv("data/test.csv")
X = np.stack([feats(Path("data") / p) for p in tr["path"]])
Xte = np.stack([feats(Path("data") / p) for p in te["path"]])
y = tr["target"].to_numpy()
skf = StratifiedKFold(5, shuffle=True, random_state=42)
oof = np.zeros(len(y)); pred = np.zeros(len(te))
for i_tr, i_va in skf.split(X, y):
    m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63,
                           n_jobs=4, verbose=-1).fit(X[i_tr], y[i_tr])
    oof[i_va] = m.predict_proba(X[i_va])[:, 1]
    pred += m.predict_proba(Xte)[:, 1] / 5
print("OOF AUC", roc_auc_score(y, oof))
pl.DataFrame({"id": te["id"], "target": pred}).write_csv("submissions/submission.csv")
```

Feature extraction is the slow part: ~10–30 s per 1k clips of 3 s. The GBDT is
seconds. This baseline often reaches 80–90% of the best score on small audio sets.

---

## 3. Best CPU model

1. **Spectral statistics + LightGBM** (above): default for N > 2k or <60 min.
2. **Scratch `AudioConformerLite`** (template default, ~0.4M params) on 3 s
   log-mel with delta/delta-delta, PCEN, SpecAugment, waveform mixup. Viable for
   N ≤ 2k, 2–3 folds, 4–8 epochs.
3. **`AudioResNet1D` / `Spec2DResNetScratch`**: similar cost; ResNet1D is faster,
   Spec2D benefits from 2D structure but is heavier.
4. **Blend** spectral-GBDT and conformer OOF (rank or logit). Template does this
   with `rust_blend_oofs`.

```bash
# Scratch conformer, CPU budget
$PY lmml_master.py solve lvl_08 -- \
  --mode audio_clf --data-dir data --arch conformer \
  --sr 16000 --duration-s 3.0 --n-mels 64 --epochs 4 --folds 2 \
  --batch-size 16 --tta-crops 2 --no-pretrained

# Pure spectral/forensic modes (fast)
$PY lmml_master.py solve lvl_08 -- --mode audio_forensics --data-dir data
$PY lmml_master.py solve lvl_08 -- --mode genai_rag --rag-corpus data/corpus.jsonl --rag-queries data/queries.csv

# ASR (only if fair play allows pretrained ASR); gate must be opened explicitly
$PY lmml_master.py solve lvl_08 -- \
  --mode asr --audio data/clip.wav --whisper-size base --language en --pretrained
# or, with a local CTranslate2 Whisper model (no download):
#   --mode asr --audio data/clip.wav --whisper-model-dir /path/to/ct2-model

# Validate the file before upload
$PY lmml_master.py check lvl_08 --sample lvl_08/data/sample_submission.csv
```

**Path rule:** `lmml_master.py solve` runs with `cwd = lvl_08/`, so template
arguments use `data/...` while `check --sample` runs from the repo root and uses
`lvl_08/data/...`.

Do not use `--arch ssl_probe` or `--pretrained` unless fair play explicitly allows
pretrained speech models. `ssl_probe` now **does** honor the Fair-Play gate:
without `--pretrained` it raises `PermissionError` instead of downloading
wav2vec/HuBERT. It is also slow on CPU. If in doubt, use `conformer`/`resnet1d`.
`--mode all` is the offline demo suite, not a production mode.

---

## 4. Feature engineering that pays on CPU

- Log-mel (64 or 128 mels), then per-channel mean/std/max/min/percentiles 10/50/90
  over time; same for delta and delta-delta.
- Spectral: centroid, bandwidth, rolloff, flatness, contrast (7 bands), chroma,
  zero-crossing rate, RMS, tempo, onset rate.
- Duration, silence ratio, leading/trailing silence, peak/RMS dBFS, clipping rate.
- Per-sample RMS normalization and bandpass 80–7600 Hz for speech.
- Segment-level features (split long clips into 1–3 s windows, aggregate mean/std/max)
  if labels are clip-level; **split by recording** to avoid leakage.
- For spoof/deepfake: high-frequency energy, phase discontinuity, spectral flatness,
  codec artifacts, resampling traces.

---

## 5. Validation

- Split by **speaker/recording/site**, never by random chunk. Chunks from the same
  file leak heavily.
- `StratifiedGroupKFold(groups=recording_id)`; for multi-label, use iterative
  stratification or at least check per-class fold counts.
- For ASR: report WER/CER on a held-out set; normalize case/punctuation before
  scoring.
- For RAG: evaluate retrieval recall@k and answer exact-match separately; do not
  tune on the final answer only.
- Check sample rate and channel count consistency between train and test; resample
  test to train rate if needed.

---

## 6. Common pitfalls

| Pitfall | Symptom | Fix |
| :--- | :--- | :--- |
| Chunk leakage | CV 0.99, LB 0.7 | Group split by file/speaker |
| Loudness bias | Model learns recording gain | RMS normalize, per-sample standardization |
| Padding silence | Short clips misclassified | Pad/crop to fixed length; mask padding |
| Wrong sample rate | Train/test distribution gap | Resample everything to 16 kHz |
| `--arch ssl_probe` without permission | `PermissionError` (Fair-Play gate) | Use scratch conformer or spectral probe; only pass `--pretrained` if allowed |
| ASR without opening the pretrained gate | `PermissionError: faster-whisper downloads pretrained weights...` | Pass `--pretrained` (if allowed) or `--whisper-model-dir`; `faster-whisper` is installed |
| External LLM API enabled by default | Disqualification risk | Omit `--allow-external-api` (template default is off) |
| 8 epochs × 3 folds on 5k clips | 3+ hours | Spectral probe or 2 folds/4 epochs |

---

## 7. Snippets

### 7.1 Load and normalize audio robustly

```python
import librosa, numpy as np
y, sr = librosa.load(path, sr=16000, mono=True)
y = y[: int(3.0 * sr)]
y = np.pad(y, (0, max(0, int(3.0 * sr) - len(y))))
y = y / (np.sqrt(np.mean(y ** 2)) + 1e-8)
```

### 7.2 Hybrid RAG on CPU (no external API)

```python
# BM25 lexical + SVD dense + reciprocal-rank fusion; template has this built in.
# Use --mode genai_rag --rag-corpus corpus.jsonl --rag-queries queries.csv
# For a custom version: rank_bm25 or sklearn TfidfVectorizer + TruncatedSVD,
# then RRF: score = sum(1 / (60 + rank_i)).
```

### 7.3 Audio forensics first

```bash
$PY lmml_master.py solve lvl_08 -- --mode audio_forensics --data-dir data
# Checks: RIFF chunks, trailing bytes, PCM LSB bit-planes, DTMF, Morse,
# ultrasonic energy, reversed-audio asymmetry, embedded flag strings.
```

---

## 8. Runtime on 4 cores (shared box)

| Operation | Time |
| :--- | :--- |
| Mel stats, 1k clips × 3 s | 15–40 s |
| LGB on 1k × 300 features | 5–15 s |
| Conformer training, 3 s clips, bs16 | ~12 clips/s measured → ~80 s/epoch/1k clips |
| Conformer 8 epochs × 3 folds, 1k clips | ~30–45 min |
| Conformer 8 epochs × 3 folds, 5k clips | 2.5–4 h — **do not** |
| Audio forensics scan | seconds–minutes |
| ASR `faster-whisper base` CPU (if installed) | ~0.2–0.5× real time; 1 h audio = 2–5 h |

## 9. Do not attempt without GPU

- Whisper medium/large, wav2vec2/HuBERT fine-tuning, speech SSL training.
- 224×224 spectrogram CNNs on >5k clips, multi-window 3-channel ensembles.
- LLM fine-tuning, vLLM, QLoRA, or high-volume external ASR/LLM APIs.
- Diffusion audio generation, source separation, TTS.
