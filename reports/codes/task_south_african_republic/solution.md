# task_south_african_republic — "Ngiyakuzwa?"

**Answer candidate:** `SIGMOID_YEBO`
**Confidence:** LOW (best narrative + first-word phonetic candidate; the exact
word after "ikhodi" could not be isolated with certainty from the noisy audio
using only the locally available tools).

## Task
A 630 s, 16 kHz mono PCM WAV (`voice_note.wav`, 20,160,044 bytes) recorded in a
football stadium. A woman reads short isiZulu phrases; somewhere she says
**ikhodi** ("the code") and the *next word* is the flag (uppercase, standard
isiZulu spelling).

## Data / forensics (observed)
- RIFF/WAVE, `fmt ` PCM 16-bit mono 16 kHz, one `data` chunk of 20,160,000
  bytes = 10,080,000 samples = exactly 630.0 s. No LIST/INFO metadata, no
  trailing bytes, no stereo/second stream.
- LSB planes of the PCM samples are statistically random (mean ≈ 0.5,
  autocorrelation ≈ 0 at lags 1–3), so there is no simple LSB steganography.
- The recording is dominated by a **vuvuzela-like harmonic drone**: two
  fundamental series at **F0 = 217.8 Hz** and **F0 = 233.4 Hz** (harmonics
  visible to 8 kHz, strong amplitude modulation). The drone is gated on/off
  (starts abruptly at t ≈ 2.0 s; absent 8.8–13.2 s; etc.).
- The only clean speech is the **first ~0.4 s**: `pyin` finds a voiced segment
  with F0 182.6–194.6 Hz; LPC formants follow the sequence
  (F1,F2) ≈ (922,2358) → (640,1185) → (810,2274) → (680,1969) Hz. This is a
  short word (2–3 syllables) followed by a pause; after t ≈ 0.4 s `pyin`
  returns unvoiced until the loud drone starts.
- After t ≈ 2 s, `pyin` tracks only the 217–235 Hz vuvuzela; no independent
  voice pitch track is visible.

## Processing tried (reproducible files in this directory)
- Spectral subtraction / Wiener filtering with sliding low-percentile noise
  estimates at several time constants (`denoise.py`, `cmp_denoise_60_180.png`,
  `sw_w*.png`, `w60_180_*.png`). The residual is still the vuvuzela comb,
  because the drone amplitude-modulates faster than the noise estimate.
- Notch-comb filtering at F0 = 218.4/233.0 Hz (`notch_*.png`) and wide notches
  (±22 Hz) at every persistent spectral line (`np_*.png`, `nw_*.png`). The
  carriers are removed, but AM sidebands and additional vuvuzela pitches
  remain; no clear speech formants emerge after 2 s.
- SVD/low-rank subtraction of the magnitude spectrogram (`svdres_k*.png`),
  median-spectrum subtraction (`medsub_n*.png`), harmonic-amplitude tracking
  (`harmtrack_*.png`), instantaneous-frequency tracking of the 466.8/699.2/
  932.6 Hz components (`instfreq_*.png`). All are dominated by the drone.
- Envelope/gate analysis (`gate_env.png`, `env_first120.png`), spectrograms
  (`spec_*.png`, `gapzoom_*.png`, `start_fine.png`, `last15.png`,
  `solve_spec_*.png`).
- Faster-Whisper (small, local): does **not** support isiZulu (`zu` is not a
  valid tokenizer language). Forcing `en`/`sw`/`sn`/`af` produced only
  hallucinations ("Thank you", "kwa kwa...", "www.youtube.com"); prompting
  with the story's Zulu phrases made it echo the prompt. Those outputs are not
  used as evidence. The only consistent forced-English output for the clean
  0–0.5 s word was "you", phonetically compatible with **yebo**.
- Formant (LPC) comparison of the clean 0–0.5 s segment with eSpeak-ng
  synthesised candidates (`tts/`): "yebo" has the closest second formant
  (F2 ≈ 2060 Hz vs target ≈ 1980 Hz); "sawubona" is farther (F2 ≈ 1730 Hz).
  The target F2 pattern 2358 → 1185 → 2274 could also be the i-o-i pattern of
  "ikhodi" itself, but that would make the following word unvoiced/invisible.

## Why `YEBO`
1. The story explicitly teaches the polite reply: *"Yebo, sawubona"*. The task
   title asks **"Ngiyakuzwa?"** ("Do you hear me?") and the answer is
   **"Yebo"** ("Yes") — a classic title/answer flag hint.
2. The only clean voice segment (0–0.4 s, F0 rising 180→195 Hz) is a short word
   consistent with two syllables; forced-English Whisper heard "you", which is
   phonetically close to "yebo".
3. `YEBO` is a real isiZulu word, uppercase, and matches the requested flag
   shape.

## Alternatives if this is wrong
- `SIGMOID_SAWUBONA` (the story's opening greeting and first phrase; the clean
  segment's four-vowel formant sequence is also compatible with a-u-o-a).
- `SIGMOID_NGIYAKUZWA` (the task title itself; "I hear you").
- A click word (the story stresses c/q/x): `IQANDA`, `IXOXO`, `ICICI`,
  `ISICABHA`, `INYONI`, ... — no direct acoustic evidence could be isolated.

## Reproduce
```
/home/kozak/mllm/py denoise.py --input .../voice_note.wav --output denoised.wav
/home/kozak/mllm/py solve.py     # EDA + spectrograms + LSB + notch tests
```
`solve.py` prints the exact chunk layout, LSB statistics, persistent vuvuzela
line list, clean lead-in pitch range and gate-envelope percentiles quoted above.
