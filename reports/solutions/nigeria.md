# Solution Report: task_nigeria ("The Gángan Never Lies")

## 1. Overview & Objective
- **Task ID:** `task_nigeria` (Nigeria).
- **Difficulty & Score:** Easy, 100 / 100 points.
- **Domain:** General AI / Audio Forensics & Signal Processing.
- **Output Format:** Secret flag `SIGMOID_<WORD>` in `answer.txt`.
- **Premise:**
  - An archival digitization project in Lagos revealed `challenge.wav`, recorded by a retired sound engineer featuring a Yoruba talking drum (*gángan*).
  - The engineer concealed a secret message inside the recording that must be extracted to recover the flag.

---

## 2. Key Insights & Forensic Methodology
- **Container & Metadata Triage:**
  - RIFF/WAVE header inspection confirmed a clean structure containing only two standard chunks: `fmt ` (16 bytes) and `data` (448,512 bytes) with zero trailing bytes or extraneous metadata chunks (`LIST`, `INFO`).
  - Audio specifications: Mono, 16-bit signed LE, sampling rate $f_s = 22,050\text{ Hz}$, 224,256 samples (duration: 10.17 s).
- **Steganography Negative Checks:**
  - LSB analysis across all 16 bit planes showed unbiased bit distributions (mean values 0.4988–0.5013).
  - Automated scans for embedded ASCII payloads, DTMF tones, and Morse code returned negative.
- **Spectrogram Frequency-Time Analysis:**
  - Frequency inspection identified an artificial amplitude-modulated pattern in the **3.0–3.7 kHz** band displaying visual letterforms.
  - **High-Resolution STFT:** Short-Time Fourier Transform using a Hann window with `nperseg = 1024` and `noverlap = 1008` (time resolution $\approx 0.73\text{ ms}$).
  - **Spectral Whitening:** Subtracted the temporal median from each frequency row (`mag - np.median(mag, axis=1, keepdims=True)`) to eliminate background drum acoustic energy.
  - Normalized amplitudes and applied threshold binarization to yield clean glyphs.

---

## 3. Implementation Details
The extraction pipeline is implemented across `solve.py`, `solution.md`, and scripts in `exploration/`:

1. **Audio Decoding (`read_wav` in `solve.py`):**
   - Reads raw PCM samples from `challenge.wav` and verifies header integrity.
2. **Spectrogram Generation & Filtering:**
   - Computes `scipy.signal.stft(x, fs=22050, window='hann', nperseg=1024, noverlap=1008)`.
   - Slices the frequency matrix between 2.8 kHz and 3.9 kHz.
   - Whitens rows, clamps negative values to zero, and normalizes dynamic range to $[0, 1]$.
3. **Glyph Decoding & Verification:**
   - Between $t = 0.70\text{ s}$ and $t = 9.65\text{ s}$, 11 evenly spaced capital letters are clearly rendered:
     $$\text{T}\quad\text{A}\quad\text{L}\quad\text{K}\quad\text{I}\quad\text{N}\quad\text{G}\quad\text{D}\quad\text{R}\quad\text{U}\quad\text{M}$$
   - Forms the single compound English word `TALKINGDRUM` (the direct translation of *gángan*).
   - Analysis of time-reversed audio (`spectrogram_reversed.png`) confirmed literal directional rendering (`MURDGNIKLAT`).
   - The spacing between `G` and `D` is identical to intra-character spacing, ruling out underscores or delimiters.

---

## 4. Results & Artifacts
- **Status:** PASSED
- **Score:** 100 / 100 points
- **Recovered Flag:** `SIGMOID_TALKINGDRUM`
- **Output Artifacts:**
  - `answer.txt` — contains the flag submission string;
  - `spectrogram.png` — full-range spectrum visualization;
  - `spectrogram_message.png` — isolated message frequency band;
  - `message_binary2.png` — binarized text mask for visual/OCR verification.
