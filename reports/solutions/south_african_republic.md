# Solution Report: task_south_african_republic ("Ngiyakuzwa?")

## 1. Overview & Objective
- **Task ID:** `task_south_african_republic` (South Africa / isiZulu).
- **Domain:** Audio Forensics, Noise Reduction (Spectral Subtraction), Speech Recognition (ASR), Formant Analysis.
- **Output Format:** Text string `SIGMOID_<WORD>` (uppercase Zulu word).
- **Input Data:**
  - `voice_note.wav`: Exactly 630.0s (10m 30s), 20,160,044 bytes.
  - Format: RIFF/WAVE, 16-bit PCM mono @ 16,000 Hz (10,080,000 samples), single 20,160,000-byte `data` chunk, no extra metadata or streams.
- **Premise:**
  - Recorded at a South African football stadium with loud vuvuzela drone and crowd noise.
  - A female speaker utters short phrases in isiZulu. At some point she says the keyword **"ikhodi"** ("code"), and the immediate next word is the secret flag.
- **Goal:** Suppress stadium vuvuzela noise, localize speech segments, and recover the flag word following "ikhodi".

---

## 2. Key Insights & Methodology

### 2.1 Audio Forensics & Noise Profile
- **LSB Steganography Check:** Bitplane analysis of the 3 least significant bits of 16-bit PCM samples showed mean $\approx 0.5$ and near-zero autocorrelation at lags 1–3, ruling out naive LSB payloads.
- **Spectral Analysis:** The recording is dominated by severe harmonic vuvuzela drone at two fundamentals:
  $$F_{0,1} = 217.8\text{ Hz}, \quad F_{0,2} = 233.4\text{ Hz}$$
  Harmonics extend up to 8 kHz with heavy amplitude modulation (~55 Hz AM). The drone kicks in abruptly at $t \approx 2.0\text{ s}$ with occasional dips (e.g., 8.8–13.2 s).
- **Clean Speech Window:** The only relatively uncorrupted speech segment occurs at the very start: $t \in [0.0, 0.4\text{ s}]$. The `pyin` fundamental frequency tracker identifies voiced speech with $F_0 \in [182.6, 194.6]\text{ Hz}$.

### 2.2 Denoising & Filtering
- **Spectral Subtraction (`denoise.py`):**
  - STFT with window size 1024 and hop size 256.
  - Noise estimated via a 4.0-second sliding 10th percentile filter across time (`scipy.ndimage.percentile_filter`).
  - Smoothed via `uniform_filter1d` across frequency (3 Hz) and time (1.0 s).
  - Magnitude subtraction with oversubtraction factor `over=1.5` and spectral floor `floor=0.02 * mag`, followed by phase inversion and ISTFT.
  - 6th-order Butterworth bandpass filter (120–5000 Hz) and peak normalization to 0.95 (`denoised.wav`).
- **Bandpass & Wiener Filter (`preprocess.py`):**
  - 4th-order Butterworth bandpass (300–3400 Hz) + stationary noise reduction via `noisereduce.reduce_noise(prop_decrease=0.97)` (`clean.wav`).
- **Notch-Comb Filter:** Notch filters at 218.4 Hz and 233.0 Hz and their harmonics failed to cleanly recover speech formants past $t=2.0\text{ s}$ due to dense AM sidebands.

### 2.3 Phonetic & Semantic Candidate Analysis
- **Whisper Limitations:** Faster-Whisper lacks native Zulu (`zu`) support. Forced English transcription on the clean lead-in [0, 0.5 s] yielded "you".
- **LPC Formant Trajectory:** Formant analysis on [0, 0.5 s] showed:
  $$(F_1, F_2) \approx (922, 2358) \to (640, 1185) \to (810, 2274) \to (680, 1969)\text{ Hz}$$
  Compared to eSpeak-ng synthetic samples, this closely matches **"yebo"** (synthetic $F_2 \approx 2060\text{ Hz}$ vs. original $1980\text{ Hz}$; alternative "sawubona" has $F_2 \approx 1730\text{ Hz}$).
- **Semantic Prompt Hint:** The challenge title **"Ngiyakuzwa?"** translates from Zulu as *"Do you hear me?"*. The standard affirmative response in Zulu etiquette is **"Yebo"** (*"Yes"*), forming a classic "question in title, answer in flag" pattern.

---

## 3. Implementation Modules

1. **`solve.py` (Forensics & EDA):**
   - Parses WAV RIFF headers and validates chunk offsets.
   - Tests LSB randomness across bit planes $k \in \{0, 1, 2\}$.
   - Runs global median power spectrum analysis ($N=16384$) with 201-tap median filtering (`signal.medfilt`) and peak detection (`find_peaks`, +15 dB threshold).
   - Generates spectrograms (`solve_spec_0_20.png`, `solve_spec_leadin.png`).
   - Tracks pitch with `librosa.pyin` (70–400 Hz, hop 128) and saves 4 kHz envelope RMS to `solve_gate.npz`.
2. **`denoise.py` (Percentile Spectral Subtraction):**
   - Implements custom STFT/ISTFT with sliding percentile noise estimation and bandlimiting.
3. **`preprocess.py` (Stationary Denoising):**
   - Noise profile extracted from initial 2.0s; applies `noisereduce` and 300–3400 Hz bandpass.
4. **`transcribe.py` (Chunked Whisper Pipeline):**
   - Splits `clean.wav` into 45s chunks with 6s overlap.
   - Multiprocess pool (`spawn`) running `WhisperModel(compute_type="int8")` with VAD disabled.
   - Outputs timestamps and tokens to `transcript.jsonl`.
5. **`solution.md`:**
   - Evaluates alternative candidate flags: `SIGMOID_YEBO` (affirmative match), `SIGMOID_SAWUBONA` (greeting), `SIGMOID_NGIYAKUZWA` (echo), and click-consonant tokens (`IQANDA`, `IXOXO`, `ICICI`, `ISICABHA`).

---

## 4. Results & Artifacts
- **Flag Candidate:** `SIGMOID_YEBO`
- **Confidence:** Low / Heuristic (due to extreme vuvuzela corruption past 2.0s; supported by 0–0.4s formants and title semantics).
- **Key Artifacts (`task_south_african_republic/`):**
  - `answer.txt`: Contains `SIGMOID_YEBO`.
  - `solution.md`: Technical documentation of acoustic measurements and formant comparisons.
  - `solve.py`: WAV parsing, LSB checks, peak finding, and spectrogram scripts.
  - `denoise.py`: Custom percentile spectral subtraction.
  - `preprocess.py`: `noisereduce` + bandpass pipeline.
  - `transcribe.py`: Multiprocess Whisper transcription script.
