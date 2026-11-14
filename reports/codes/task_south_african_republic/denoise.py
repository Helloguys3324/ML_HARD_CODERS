#!/usr/bin/env python3
"""Denoise the stadium voice note via spectral subtraction.

The background is a vuvuzela-like harmonic drone (~219 Hz tone, ~55 Hz AM)
plus crowd roar.  Speech is non-stationary, so a time-varying noise estimate
(low percentile of |STFT| over a sliding window) and spectral subtraction
recovers it reasonably well.
"""
import argparse
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import percentile_filter, uniform_filter1d

SR = 16000


def load(path):
    x, sr = sf.read(path, dtype="float32")
    assert sr == SR, sr
    return x


def stft(x, n=1024, hop=256):
    f, t, Z = signal.stft(x, fs=SR, nperseg=n, noverlap=n - hop,
                          window="hann", padded=False, boundary=None)
    return f, t, Z


def denoise(x, n=1024, hop=256, pct=10, win_s=4.0, over=1.5, floor=0.02,
            smooth_hz=3, smooth_t=1.0):
    f, t, Z = stft(x, n, hop)
    mag = np.abs(Z)
    # sliding low-percentile noise estimate per frequency bin
    w = max(3, int(round(win_s / (hop / SR))))
    noise = percentile_filter(mag, percentile=pct, size=(1, w), mode="nearest")
    # smooth noise estimate a little in frequency/time
    nf = max(1, int(round(smooth_hz / (SR / n))))
    nt = max(1, int(round(smooth_t / (hop / SR))))
    noise = uniform_filter1d(noise, size=nf, axis=0, mode="nearest")
    noise = uniform_filter1d(noise, size=nt, axis=1, mode="nearest")
    clean_mag = mag - over * noise
    clean_mag = np.maximum(clean_mag, floor * mag)
    Zc = clean_mag * np.exp(1j * np.angle(Z))
    _, y = signal.istft(Zc, fs=SR, nperseg=n, noverlap=n - hop,
                        window="hann", boundary=None)
    y = y[:len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y


def bandlimit(x, lo=120, hi=5000, order=6):
    sos = signal.butter(order, [lo, hi], btype="bandpass", fs=SR, output="sos")
    return signal.sosfiltfilt(sos, x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/home/kozak/mllm/MML/work/tasks/task_south_african_republic/data/voice_note.wav")
    ap.add_argument("--output", default="denoised.wav")
    ap.add_argument("--pct", type=float, default=10)
    ap.add_argument("--win", type=float, default=4.0)
    ap.add_argument("--over", type=float, default=1.5)
    ap.add_argument("--floor", type=float, default=0.02)
    ap.add_argument("--lo", type=float, default=120)
    ap.add_argument("--hi", type=float, default=5000)
    ap.add_argument("--no-bandlimit", action="store_true")
    args = ap.parse_args()
    x = load(args.input)
    y = denoise(x, pct=args.pct, win_s=args.win, over=args.over, floor=args.floor)
    if not args.no_bandlimit:
        y = bandlimit(y, args.lo, args.hi)
    # normalize
    peak = np.max(np.abs(y)) + 1e-12
    y = y / peak * 0.95
    sf.write(args.output, y, SR, subtype="PCM_16")
    print("wrote", args.output, len(y) / SR, "s")


if __name__ == "__main__":
    main()
