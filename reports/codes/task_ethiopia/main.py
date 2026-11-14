#!/usr/bin/env python3
"""The Azmari's Borrowed Melody - audio splice localisation.

A few seconds of a real recording were replaced by audio from a different
take / recording.  Find the start and end of the replacement (seconds from
the beginning of the file) and write them to ``detection.csv``.

Usage:
    python main.py test-1-input.wav

Method
------
Splicing material from another recording preserves the loudness (the editor
matched levels) but changes the acoustic fingerprint: the long-term spectral
profile and the noise-floor / spectral-valley profile of the inserted passage
differ from the host recording.  For every STFT frame we build two
level-invariant log band profiles (band energy and 20th-percentile band
noise floor), standardise them per band with robust (median/MAD) statistics,
and scan all contiguous segments for the one whose mean profile differs most
from the rest of the file.  The two boundaries are then snapped to the exact
step of the profile projection (matched filter).
"""
import sys
import csv

import numpy as np


# --------------------------------------------------------------------------
# WAV reading
# --------------------------------------------------------------------------
def read_wav(path):
    """Return (mono float64 signal, sample rate)."""
    try:
        from scipy.io import wavfile
        sr, data = wavfile.read(path)
        if data.dtype == np.uint8:
            x = (data.astype(np.float64) - 128.0) / 128.0
        elif data.dtype == np.int16:
            x = data.astype(np.float64) / 32768.0
        elif data.dtype == np.int32:
            x = data.astype(np.float64) / 2147483648.0
        else:
            x = data.astype(np.float64)
        if x.ndim > 1:
            x = x.mean(axis=1)
        return x, int(sr)
    except Exception:
        pass

    import wave
    w = wave.open(path, "rb")
    try:
        sr = w.getframerate()
        nch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    finally:
        w.close()
    if sw == 1:
        a = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif sw == 2:
        a = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v >= (1 << 23), v - (1 << 24), v)
        a = v.astype(np.float64) / float(1 << 23)
    elif sw == 4:
        a = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        a = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
    if nch > 1:
        a = a.reshape(-1, nch).mean(axis=1)
    return a.astype(np.float64), int(sr)


# --------------------------------------------------------------------------
# Frame features
# --------------------------------------------------------------------------
def band_features(x, sr):
    """Log band energies, log band noise floors and frame activity."""
    n = len(x)
    win = int(2 ** round(np.log2(max(512.0, 0.093 * sr))))
    hop = max(1, win // 8)
    nfr = max(1, (n - win) // hop + 1)
    if nfr < 3:
        return None

    freq = np.fft.rfftfreq(win, 1.0 / sr)
    hann = np.hanning(win)

    edges = np.geomspace(40.0, 0.45 * sr, 24)
    masks = [np.where((freq >= edges[i]) & (freq < edges[i + 1]))[0]
             for i in range(len(edges) - 1)]
    nb = len(masks)

    E = np.empty((nfr, nb), dtype=np.float64)
    F = np.empty((nfr, nb), dtype=np.float64)

    chunk = 512
    for s in range(0, nfr, chunk):
        e = min(nfr, s + chunk)
        seg = x[s * hop: (e - 1) * hop + win]
        frames = np.lib.stride_tricks.sliding_window_view(seg, win)[::hop][: e - s]
        spec = np.abs(np.fft.rfft(frames * hann, axis=1))
        pw = spec * spec + 1e-24
        for bi, idx in enumerate(masks):
            if idx.size == 0:
                E[s:e, bi] = 1e-20
                F[s:e, bi] = 1e-20
            else:
                sub = pw[:, idx]
                E[s:e, bi] = sub.sum(axis=1)
                F[s:e, bi] = np.percentile(sub, 20, axis=1)

    t = (np.arange(nfr) * hop + win * 0.5) / sr

    # Frames whose energy is far below the file median (silence, fades,
    # dropouts) must not be mistaken for an inserted passage.
    e_log = np.log10(E.sum(axis=1) + 1e-20)
    active = e_log > (np.median(e_log) - 2.0)          # -20 dB re median
    if active.sum() < max(10, int(0.1 * nfr)):
        active = np.ones(nfr, dtype=bool)

    return {"t": t, "E": E, "F": F, "fr": sr / hop, "active": active}


def normalise_profile(X):
    """Level-invariant log profile -> per-band robust z-score."""
    L = np.log10(X + 1e-18)
    L -= L.mean(axis=1, keepdims=True)
    med = np.median(L, axis=0)
    mad = np.median(np.abs(L - med), axis=0) + 1e-3
    return np.clip((L - med) / mad, -12.0, 12.0)


# --------------------------------------------------------------------------
# Segment search / boundary refinement
# --------------------------------------------------------------------------
def smooth_frames(Z, frames):
    if frames <= 1:
        return Z
    k = np.ones(frames) / float(frames)
    out = np.empty_like(Z)
    for b in range(Z.shape[1]):
        out[:, b] = np.convolve(Z[:, b], k, mode="same")
    return out


def scan_segment(Z, act, fr, nfr, margin, minlen, maxlen, step,
                 i_lo=None, i_hi=None, j_lo=None, j_hi=None):
    """Contiguous segment whose mean profile differs most from the rest.

    ``act`` marks frames that carry signal (not silence / dropouts).  Silent
    stretches are not allowed to *be* the detected replacement, but they are
    still used when averaging the profiles.
    """
    nb = Z.shape[1]
    Zs = smooth_frames(Z, max(3, int(round(0.25 * fr))))
    cs = np.concatenate([np.zeros((1, nb)), np.cumsum(Zs, axis=0)], axis=0)
    cn = np.concatenate([[0.0], np.cumsum(act.astype(np.float64))])
    total = cs[-1]
    tot_n = cn[-1]

    stepn = max(1, int(round(step * fr)))
    marg = max(1, int(round(margin * fr)))
    mn = max(3, int(round(minlen * fr)))
    mx = max(mn + 1, int(round(maxlen * fr)))

    lo_i = marg if i_lo is None else max(1, int(i_lo))
    hi_i = nfr - marg if i_hi is None else min(nfr - 1, int(i_hi))
    g_jlo = marg if j_lo is None else max(marg, int(j_lo))
    g_jhi = nfr - marg if j_hi is None else min(nfr - marg, int(j_hi))

    best = (-1.0, 0, nfr - 1)
    for i in range(lo_i, hi_i, stepn):
        jlo = max(i + mn, g_jlo)
        jhi = min(min(nfr - marg, i + mx), g_jhi + 1)
        if jhi <= jlo:
            continue
        js = np.arange(jlo, jhi, stepn)
        if js.size == 0:
            continue
        segsum = cs[js] - cs[i]
        L = (js - i).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            mseg = segsum / L[:, None]
            mout = (total - segsum) / (nfr - L)[:, None]
        dd = mseg - mout
        score = np.sqrt(np.mean(dd * dd, axis=1)) * np.sqrt(L)
        seg_n = cn[js] - cn[i]
        ok = (seg_n >= 0.6 * L) & ((tot_n - seg_n) >= 0.6 * (nfr - L))
        score = np.where(ok, score, -1.0)
        k = int(np.argmax(score))
        if score[k] > best[0]:
            best = (float(score[k]), int(i), int(js[k]))
    return best


def refine_boundaries(Z, act, fr, i, j, nfr, search=1.0, half=0.6):
    """Snap segment edges to the strongest step of the profile projection."""
    Zs = smooth_frames(Z, max(3, int(round(0.12 * fr))))
    mseg = Zs[i:j].mean(axis=0)
    mout = np.concatenate([Zs[:i], Zs[j:]]).mean(axis=0)
    d = mseg - mout
    nd = np.sqrt(np.mean(d * d)) + 1e-9
    p = Zs @ (d / nd)

    W = max(2, int(round(half * fr)))
    sw = max(1, int(round(search * fr)))

    def step_at(tau):
        a = p[max(0, tau - W):tau]
        b = p[tau:min(nfr, tau + W)]
        return (b.mean() if b.size else 0.0) - (a.mean() if a.size else 0.0)

    lo = max(W, i - sw)
    hi = min(nfr - W, i + sw)
    if hi > lo:
        vals = [step_at(tau) for tau in range(lo, hi + 1)]
        st = lo + int(np.argmax(vals))
    else:
        st = i

    lo = max(W, j - sw)
    hi = min(nfr - W, j + sw)
    if hi > lo:
        vals = [step_at(tau) for tau in range(lo, hi + 1)]
        en = lo + int(np.argmin(vals))
    else:
        en = j

    return st, en


# --------------------------------------------------------------------------
# Main detector
# --------------------------------------------------------------------------
def detect(path):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x, sr = read_wav(path)
        x = np.nan_to_num(np.asarray(x, dtype=np.float64))
        x = x - np.mean(x)
    dur = len(x) / float(sr)

    if dur < 1.0 or np.max(np.abs(x)) <= 0.0:
        return 0.0, min(dur, 1.0)

    fe = band_features(x, sr)
    if fe is None:
        return 0.0, dur * 0.5

    Z = np.concatenate([normalise_profile(fe["E"]),
                        normalise_profile(fe["F"])], axis=1)
    act = fe["active"]
    fr = fe["fr"]
    nfr = len(fe["t"])
    t = fe["t"]

    # The replacement is "a few seconds" long and sits inside the recording.
    margin = min(1.0, max(0.2, 0.03 * dur))
    minlen = max(1.2, min(2.0, 0.03 * dur))
    maxlen = min(25.0, max(3.0, 0.65 * dur))
    step = 0.25 if dur <= 120.0 else 0.4

    score, i, j = scan_segment(Z, act, fr, nfr, margin, minlen, maxlen, step)

    # Local re-scan: strip host material the length prior may have attached
    # around the true replacement.  Only sub-segments of the coarse estimate
    # are considered, with a finer grid.
    pad = max(2, int(round(0.4 * fr)))
    trim = max(2, int(round(2.5 * fr)))
    score2, i2, j2 = scan_segment(
        Z, act, fr, nfr, margin, 1.0, maxlen,
        step=0.08,
        i_lo=max(1, i - pad),
        i_hi=min(nfr - 1, i + trim),
        j_lo=max(1, j - trim),
        j_hi=min(nfr, j + pad),
    )
    if score2 > 0 and (j2 - i2) >= max(3, int(0.4 * (j - i))):
        i, j = i2, j2

    # Guard: if the best segment reaches the search margin, the replacement
    # may start or end at the very edge of the file.
    stepn = int(round(step * fr))
    at_edge_start = i <= int(round(margin * fr)) + stepn
    at_edge_end = j >= nfr - int(round(margin * fr)) - stepn

    st, en = refine_boundaries(Z, act, fr, i, j, nfr)

    if at_edge_start and st > i + int(round(0.8 * fr)):
        st = i
    if at_edge_end and en < j - int(round(0.8 * fr)):
        en = j

    start_s = float(t[max(0, min(nfr - 1, st))])
    end_s = float(t[max(0, min(nfr - 1, en))])
    start_s = max(0.0, min(start_s, dur))
    end_s = max(0.0, min(end_s, dur))
    if end_s < start_s:
        start_s, end_s = end_s, start_s
    return start_s, end_s


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("usage: python main.py <input.wav>\n")
        return 1
    start_s, end_s = detect(sys.argv[1])
    with open("detection.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["start_s", "end_s"])
        w.writerow(["{:.3f}".format(start_s), "{:.3f}".format(end_s)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
