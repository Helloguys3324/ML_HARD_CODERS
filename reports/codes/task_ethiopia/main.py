#!/usr/bin/env python3
"""
The Azmari's Borrowed Melody (task_ethiopia) - LMML 2026

Detects the boundaries (start_s, end_s) of a spliced-in segment inside a mono/stereo
WAV recording, using only numpy + scipy (stdlib `csv` for output, mirroring the
official example harness contract):

    python main.py test-1-input.wav      -> writes detection.csv  (start_s,end_s)

Method
------
A splice replaces a few seconds of a *real* recording with a phrase taken from a
*different* take.  Two takes of the same performer are never perfectly identical in
the spectral (timbre / room-tone / recording-chain) domain, even when the melodic
content is comparable; but *within* one take all frames share the same timbre.

The splice is therefore the only contiguous region that is simultaneously

  * FOREIGN     — each frame is dissimilar to the frames far away from it (low
                  ``ext`` self-similarity), and
  * COHERENT    — the frames inside it are similar to one another (high ``loc``
                  self-similarity), because they come from one foreign take.

A natural change of playing technique can be foreign but is not coherent; a long
held note is coherent but not foreign.  Only the splice scores high on both.

Concretely:

  1. log-mel spectrogram -> DCT -> 13 MFCC per frame (timbre),
  2. gate out silent frames (energy far below the recording's own median),
  3. per-dimension mean-centre + per-frame L2-normalise the active frames,
  4. ``ext[i]`` = mean cosine similarity of frame i to frames > ext_sec away,
     ``loc[i]`` = mean cosine similarity to frames within loc_sec (minus a tiny
     self tap), both computed with the O(T*d) cumsum trick (no T x T matrix),
  5. ``contrast = loc - ext``, smoothed -> the splice is the contiguous run that
     sticks out as a positive outlier (median + k*MAD, largest integrated excess),
     with interior runs (splices live *inside* the recording) preferred over
     anything touching the file edges,
  6. refine the two edges with the (crisper) ``ext`` "foreignness" valley, capped
     to a short neighbourhood around each rough edge so it can never run away.

No external/pretrained models, no internet.  Outputs exactly one CSV file.
"""

from __future__ import annotations

import csv
import sys

import numpy as np

try:  # scipy >= 1.4
    from scipy.fft import dct
except ImportError:  # much older scipy
    from scipy.fftpack import dct

from scipy.io import wavfile
from scipy.signal import savgol_filter, stft


def mel_filterbank(n_fft: int, sr: int, n_mels: int = 40) -> np.ndarray:
    """Triangular mel filterbank, shape (n_mels, n_fft//2 + 1)."""
    fmin, fmax = 0.0, sr / 2.0
    mel = lambda hz: 2595.0 * np.log10(1.0 + hz / 700.0)
    invmel = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    pts = invmel(np.linspace(mel(fmin), mel(fmax), n_mels + 2))
    bins = np.clip(np.floor((n_fft + 1) * pts / sr).astype(int), 0, n_fft // 2)

    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(n_mels):
        l, c, r = bins[m], bins[m + 1], bins[m + 2]
        if c > l:
            fb[m, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c:
            fb[m, c:r] = (r - np.arange(c, r)) / (r - c)
    return fb


def _features(data: np.ndarray, sr: int, n_fft: int, hop: int,
              n_mels: int, n_mfcc: int, silence_db: float):
    """Return (feats, t, S, active)."""
    _, t, Z = stft(data, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, window="hann")
    S = np.abs(Z)
    fb = mel_filterbank(n_fft, sr, n_mels)
    logmel = np.log(fb @ S + 1e-10)
    feats = dct(logmel, axis=0, type=2, norm="ortho")[:n_mfcc]

    # energy gate: silence / room-tone frames are not music
    E = (S ** 2).sum(axis=0)
    EdB = 10.0 * np.log10(E + 1e-14)
    active = EdB > (np.median(EdB) - silence_db)
    if active.sum() < 3:
        active = np.ones(feats.shape[1], dtype=bool)
    return feats, t, S, active


def _window_sum(Fn: np.ndarray, w: int):
    """Per-frame sum of neighbour dot products within +/-w, shape (T,)."""
    T = Fn.shape[1]
    C = np.hstack([np.zeros((Fn.shape[0], 1)), np.cumsum(Fn, axis=1)])
    idx = np.arange(T)
    lo = np.clip(idx - w, 0, T - 1)
    hi = np.clip(idx + w + 1, 0, T)
    return (Fn * (C[:, hi] - C[:, lo])).sum(axis=0)


def find_splice(data: np.ndarray, sr: int,
                n_fft: int = 2048, hop: int = 512,
                n_mels: int = 40, n_mfcc: int = 13,
                silence_db: float = 25.0,
                ext_sec: float = 1.5,
                loc_sec: float = 3.0,
                small_sec: float = 0.15,
                smooth_sec: float = 0.75,
                k_factor: float = 2.0,
                min_len_sec: float = 0.3,
                refine_sec: float = 0.75):
    """Return (start_s, end_s) of the spliced segment, in seconds."""
    hop_s = hop / sr
    feats, t, S, active = _features(data, sr, n_fft, hop, n_mels, n_mfcc, silence_db)
    T = feats.shape[1]

    # ---- timbre vectors (centred + unit-norm over ACTIVE frames only) ----
    F = feats - feats[:, active].mean(axis=1, keepdims=True)
    Fn = np.zeros_like(F)
    nrm = np.linalg.norm(F[:, active], axis=0) + 1e-12
    Fn[:, active] = F[:, active] / nrm[None, :]

    # ---- active-frame counts (for windowed averages) ----
    ac = np.hstack([0.0, np.cumsum(active.astype(float))])
    na = active.sum()
    idx = np.arange(T)

    def wcnt(w: int):
        lo = np.clip(idx - w, 0, T - 1)
        hi = np.clip(idx + w + 1, 0, T)
        return ac[hi] - ac[lo]

    G = Fn.sum(axis=1)                     # (d,)
    S_all = Fn.T @ G                       # sum over ALL frames, O(T*d)

    ew = int(round(ext_sec / hop_s))
    lw = int(round(loc_sec / hop_s))
    sw = int(round(small_sec / hop_s))

    # foreignness: mean similarity to frames farther than ext_sec away
    ext = (S_all - _window_sum(Fn, ew)) / np.maximum(na - wcnt(ew), 1.0)
    # coherence: mean similarity to frames within loc_sec (minus self tap)
    loc = (_window_sum(Fn, lw) - _window_sum(Fn, sw)) / \
          np.maximum(wcnt(lw) - wcnt(sw), 1.0)

    contrast = loc - ext

    # ---- smooth contrast only along the active axis ----
    cont = np.full(T, np.nan)
    ai = np.where(active)[0]
    ca = contrast[ai]
    wlen = max(3, int(round(smooth_sec / hop_s)) | 1)
    if len(ca) > wlen:
        ca = savgol_filter(ca, wlen, min(3, wlen - 1))
    cont[ai] = ca

    # ---- robust positive-outlier threshold ----
    vc = cont[ai]
    med = np.median(vc)
    mad = np.median(np.abs(vc - med)) + 1e-12
    thr = med + k_factor * mad

    def collect(thr_):
        runs = []
        i = 0
        high = cont > thr_
        while i < T:
            if high[i]:
                j = i
                while j < T and high[j]:
                    j += 1
                a, b = t[i], t[j - 1]
                if (b - a) >= min_len_sec:
                    # integrated excess above threshold
                    deficit = float((cont[i:j] - thr_).sum())
                    # interior preference: a splice sits inside native material
                    edge_frac = max(min(a, 1.5) + min((t[-1] + hop_s) - b, 1.5), 0.0) / 3.0
                    runs.append((a, b, deficit * (1.0 - 0.75 * edge_frac)))
                i = j
            else:
                i += 1
        return runs

    runs = collect(thr)
    if not runs:                              # relax threshold once
        runs = collect(med + max(0.25, k_factor - 0.5) * mad)

    if not runs:                              # extreme fallback: contrast peak
        kp = int(np.nanargmax(cont))
        a, b = t[kp] - 1.0, t[kp] + 1.0
    else:
        runs.sort(key=lambda r: -r[2])
        a, b = runs[0][0], runs[0][1]

    # ---- refine edges with the crisper "foreignness" valley (ext),
    #      capped to a short neighbourhood so it cannot run away ----
    ve = ext[ai]
    me = np.median(ve)
    mde = np.median(np.abs(ve - me)) + 1e-12
    ethr = me - 1.5 * mde
    lowe = ext < ethr

    rw = int(round(refine_sec / hop_s))
    ia = int(np.argmin(np.abs(t - a)))
    ib = int(np.argmin(np.abs(t - b)))
    la = ia
    for _ in range(rw):
        if la > 0 and lowe[la - 1]:
            la -= 1
        else:
            break
    rb = ib
    for _ in range(rw):
        if rb < T - 1 and lowe[rb + 1]:
            rb += 1
        else:
            break

    return float(round(t[la], 3)), float(round(t[rb], 3))


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python main.py <input.wav>")

    input_file = sys.argv[1]
    sr, raw = wavfile.read(input_file)

    # --- normalize to float64 mono in [-1, 1] ---
    orig_dtype = raw.dtype
    if raw.ndim > 1:
        raw = raw.mean(axis=1)          # downmix stereo to mono
    x = raw.astype(np.float64)
    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        x = x / max(float(info.max), 1.0)
    else:                               # float WAV: assume [-1,1], guard range
        m = float(np.abs(x).max())
        if m > 1.0:
            x = x / m

    start_s, end_s = find_splice(x, sr)

    with open("detection.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_s", "end_s"])
        w.writerow([f"{start_s:.3f}", f"{end_s:.3f}"])


if __name__ == "__main__":
    main()