# task_nigeria — The Gángan Never Lies

**Answer candidate:** `SIGMOID_TALKINGDRUM`
**CONFIDENCE: high**

## TL;DR

`challenge.wav` is a plain 16-bit PCM mono WAV (22050 Hz, 10.17 s) whose
3.0–3.7 kHz band carries a **hidden text message drawn directly into the
spectrogram**. Whitening the band and stretching the contrast makes it read,
letter by letter, across 0.7–9.6 s:

```
T A L K I N G D R U M
```

This is one continuous word (no wider gap between G and D than between the
other letters), and it matches the story's theme exactly: a *gángan* is the
Yoruba **talking drum**. Flag: **`SIGMOID_TALKINGDRUM`**.

## Evidence

### 1. Container is clean — the secret is in the samples

RIFF walk finds only the mandatory chunks, no metadata / LIST / INFO payload:

```
WAV chunks: [('fmt ', 16, 12), ('data', 448512, 36)]
format: 1 ch, 16 bit, 22050 Hz, 224256 samples, 10.17 s
file size 448556 = 44-byte header + 448512 data bytes
```

### 2. Spectrogram reveals the message

Plain STFT of the whole file already shows a bright band of letter shapes at
~3.2–3.5 kHz: `spectrogram.png`.

* STFT `nperseg=1024`, `noverlap=1008` (≈0.73 ms hop), Hann window.
* Take 2800–3900 Hz, subtract the per-frequency time median (removes the flat
  hiss), light Gaussian smoothing → `spectrogram_message.png`.
* The rendered glyphs are unmistakably **T-A-L-K-I-N-G-D-R-U-M**; the clean
  binary version is `message_binary2.png`.
* The message starts at ≈0.70 s and ends at ≈9.65 s; 11 letters with uniform
  letter spacing → the hidden payload is the single word `TALKINGDRUM`.
  In particular the G→D gap is the same width as the other inter-letter gaps
  (no word-space), and the flag format is `SIGMOID_<WORD>` (one word), so the
  underscore form is `SIGMOID_TALKINGDRUM`, not `SIGMOID_TALKING_DRUM`.
* Reversing the audio mirrors the same text (`MURDGNIKLAT`), confirming the
  content is a forward-written word (`spectrogram_reversed.png`).

### 3. Negative checks (why it is not something else)

| Check | Result |
|---|---|
| Extra RIFF chunks / trailing bytes | none — header ends at offset 44 |
| LSB stego (all 16 bit planes) | bit means 0.4988–0.5013; no ASCII / no payload |
| LSB as 2-D bitmap (widths 512/586/672/1024) | pure dither noise, no image |
| Reversed audio | same word mirrored, no second message |
| Morse / DTMF / OOK in other bands | none: 0–1.2 kHz and 4–11 kHz are flat noise |
| SSTV / image modes | none |

The only structured, low-entropy signal in the file is the spectrogram text in
the 3 kHz band.

### 4. Thematic cross-check

The task title is *The Gángan Never Lies*; the story describes the gángan as
the Yoruba **talking drum** whose pitch imitates speech. The extracted word
`TALKINGDRUM` is therefore the semantically expected payload, not noise.

## Reproduction

```bash
cd /home/kozak/mllm/MML/work/solutions/task_nigeria
/home/kozak/mllm/py solve.py
```

`solve.py` re-parses the WAV chunks, runs the LSB/bit-plane null checks,
computes the whitened STFT and regenerates:

* `spectrogram.png` — full-band view with the message
* `spectrogram_message.png` — clean upright rendering (readable text)
* `message_binary2.png` — high-contrast binary rendering of the glyphs

and prints `FLAG = SIGMOID_TALKINGDRUM`.

## Files

| File | Purpose |
|---|---|
| `answer.txt` | `SIGMOID_TALKINGDRUM` |
| `solve.py` | reproducible solver + null checks + figures |
| `spectrogram.png` | 1.5–5 kHz view: message is clearly visible |
| `spectrogram_message.png` | whitened message band, upright and legible |
| `message_binary2.png` | high-contrast binary glyph rendering |
| `spectrogram_reversed.png` | reversed-audio control (mirrored text) |
| `spectrogram_low.png`, `spectrogram_high.png` | other bands — noise only |
| `lsb_w*.png` | LSB bit-plane images — no hidden bitmap |
| `exploration/` | scratch scripts kept for the audit trail |

## Confidence justification

The word is legible without ambiguity in three independent renderings
(different STFT window sizes and contrast mappings), the glyph sequence has
exactly 11 letters matching TALKINGDRUM, no competing structured signal exists
in any other band or in the sample LSBs, and the word is the literal theme of
the task. Confidence: **high**.
