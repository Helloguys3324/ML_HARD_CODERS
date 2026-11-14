# Solution Report: task_ethiopia ("The Azmari's Borrowed Melody")

## 1. Problem Overview
- **Task & Track:** Medium (200 pts). Archival solo performances of Ethiopian azmari musicians (masenqo lute or washint flute) contain a seamless audio splice inserted from another take without audible volume jumps or click artifacts.
- **Objective:** Detect the exact temporal boundaries (`start_s`, `end_s` in seconds) of the inserted fragment across 3 hidden test recordings (`test-1`, `test-2`, `test-3`).
- **Submission Format & Test Protocol:**
  - Standalone script `main.py` executed as: `python main.py <input.wav>`.
  - Generates `detection.csv` in the working directory:
    ```csv
    start_s,end_s
    00.0,00.0
    ```
  - Success criteria: Absolute boundary errors $|start - true\_start| \le 1.5\,\text{s}$ and $|end - true\_end| \le 1.5\,\text{s}$ on all 3 test files simultaneously.

## 2. Key Insight & Methodology
While RMS loudness and transient edges were matched during splicing, the recording environment's acoustic transfer function and background noise floor valley profiles differ between takes.

1. **Spectral Profiling:** Short-Time Fourier Transform (STFT) extracted across 24 logarithmically spaced frequency bands (40 Hz to $0.45 \cdot f_s$). Per-band features include total power ($E$) and the 20th percentile power ($F$).
2. **Robust Standardization:** Per-frame spectral log-energies are centered relative to the frame mean (cancelling overall loudness variations) and standardized using median and Median Absolute Deviation (MAD), clipped to $[-12, 12]$. Features are stacked into $Z \in \mathbb{R}^{N_{\text{frames}} \times 46}$.
3. **Cumulative Sum Segment Scanning (`scan_segment`):**
   - Scans candidate segments $[i, j]$ (duration 1.2s to 25s) with active frame ratio $\ge 60\%$.
   - Maximizes the length-weighted Euclidean distance between segment mean $m_{\text{seg}}$ and remainder mean $m_{\text{out}}$ using prefix sums (`cumsum`):
     $$\text{Score}(i, j) = \sqrt{\frac{1}{B}\sum_{b=1}^B (m_{\text{seg}, b} - m_{\text{out}, b})^2} \cdot \sqrt{j - i}$$
4. **Boundary Refinement (`refine_boundaries`):**
   - Projects frames onto difference vector $d = m_{\text{seg}} - m_{\text{out}}$ (matched filtering).
   - Locates start and end transitions by maximizing local step gradients of the projection curve.

## 3. Implementation Details (`main.py`)
- **Audio Loading (`read_wav`):** `scipy.io.wavfile` with fallback to standard library `wave`. Converts stereo to mono and normalizes integer bit depths (`uint8`, `int16`, `int24`, `int32`).
- **STFT Parameters:** Window size $win \approx 0.093 \cdot f_s$ (power-of-two, min 512 samples) with hop size $win // 8$. Activity mask filters deep silence ($< -20\,\text{dB}$ relative to median energy).
- **Inference Pipeline:** Computes $Z$, scans segments at 0.25s intervals (0.4s for longer audio), refines boundaries to millisecond precision, and writes `detection.csv`.

## 4. Results & Artifacts
- **Validation:** On `train-input.wav`, detected boundaries matched ground truth from `train-output.csv` within tens of milliseconds (well within $\pm 1.5\,\text{s}$).
- **Platform Status:** 15 submission attempts were logged (`teamStatus: "rejected"` in `meta.json`); the finalized standalone audio forensic detector is preserved in `main.py`.
- **Files:** `main.py`.
