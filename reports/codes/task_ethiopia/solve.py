#!/usr/bin/env python3
"""
Development/validation script for task_ethiopia ("The Azmari's Borrowed Melody").

Loads the training recording, runs the same MFCC self-similarity splice detector
used by `main.py`, compares against the ground-truth `train-output.csv`, and
renders a diagnostic figure (spectrogram + similarity profile with the detected
and true boundaries marked).

Run:
    .venv/bin/python solve.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from main import find_splice  # noqa: E402
from scipy.io import wavfile  # noqa: E402
from scipy.signal import savgol_filter, stft  # noqa: E402


def main() -> None:
    data_dir = HERE / "data"
    wav_path = data_dir / "train-input.wav"
    csv_path = data_dir / "train-output.csv"

    sr, raw = wavfile.read(wav_path)
    orig_dtype = raw.dtype
    if raw.ndim > 1:
        raw = raw.mean(axis=1)
    x = raw.astype(np.float64)
    if np.issubdtype(orig_dtype, np.integer):
        x = x / max(float(np.iinfo(orig_dtype).max), 1.0)

    # ground truth
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    gt_start, gt_end = float(rows[1][0]), float(rows[1][1])

    # detection
    start_s, end_s = find_splice(x, sr)

    print("=" * 64)
    print("task_ethiopia — splice detection (train)")
    print("=" * 64)
    print(f"  detected : {start_s:.3f} .. {end_s:.3f} s  (dur {end_s - start_s:.3f} s)")
    print(f"  truth    : {gt_start:.3f} .. {gt_end:.3f} s")
    print(f"  error    : start {abs(start_s - gt_start) * 1000:.1f} ms, "
          f"end {abs(end_s - gt_end) * 1000:.1f} ms")
    ok = abs(start_s - gt_start) <= 1.5 and abs(end_s - gt_end) <= 1.5
    print(f"  PASS     : {'YES' if ok else 'NO'}  (tolerance 1.5 s)")

    # ---- build the diagnostic figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_fft, hop = 2048, 512
    f, t, Z = stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, window="hann")
    S = 20 * np.log10(np.abs(Z) + 1e-9)

    # recompute the (energy-gated) self-similarity profile for the plot
    from main import _mfcc
    feats, tf, Smag = _mfcc(x, sr, n_fft, hop, 40, 13)
    T = feats.shape[1]
    E = (Smag ** 2).sum(axis=0)
    EdB = 10.0 * np.log10(E + 1e-14)
    active = EdB > (np.median(EdB) - 25.0)
    if active.sum() < 3:
        active = np.ones(T, dtype=bool)
    F = feats - feats[:, active].mean(axis=1, keepdims=True)
    Fn = np.zeros_like(F)
    normA = np.linalg.norm(F[:, active], axis=0) + 1e-12
    Fn[:, active] = F[:, active] / normA[None, :]
    G = Fn.sum(axis=1)
    S_all = Fn.T @ G
    d = Fn.shape[0]
    C = np.hstack([np.zeros((d, 1)), np.cumsum(Fn, axis=1)])
    hop_s = hop / sr
    excl = max(1, int(round(1.0 / hop_s)))
    idx = np.arange(T)
    lo = np.clip(idx - excl, 0, T - 1)
    hi = np.clip(idx + excl + 1, 0, T)
    local = (Fn * (C[:, hi] - C[:, lo])).sum(axis=0)
    ac = np.hstack([0.0, np.cumsum(active.astype(float))])
    denom = np.maximum(ac.sum() - (ac[hi] - ac[lo]), 1.0)
    prof = (S_all - local) / denom
    act_idx = np.where(active)[0]
    pa = prof[act_idx]
    wlen = max(3, int(round(0.9 / hop_s)) | 1)
    if len(pa) > wlen:
        pa = savgol_filter(pa, wlen, min(3, wlen - 1))
    prof_plot = np.full(T, np.nan)
    prof_plot[act_idx] = pa
    va = prof_plot[act_idx]
    med = np.median(va)
    mad = np.median(np.abs(va - med)) + 1e-12
    thr = med - 1.5 * mad

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    im = ax.imshow(S, aspect="auto", origin="lower", cmap="magma",
                   extent=[t[0], t[-1], 0, sr / 2], vmin=-100, vmax=-20)
    ax.set_ylim(0, min(sr // 2, 11025))
    ax.axvline(gt_start, color="lime", lw=2, ls="--", label="ground truth")
    ax.axvline(gt_end, color="lime", lw=2, ls="--")
    ax.axvline(start_s, color="cyan", lw=2, label="detected")
    ax.axvline(end_s, color="cyan", lw=2)
    ax.axvspan(start_s, end_s, color="cyan", alpha=0.08)
    ax.set_ylabel("freq (Hz)")
    ax.set_title("train-input.wav — spectrogram with splice boundaries")
    ax.legend(loc="upper right", framealpha=0.8)
    fig.colorbar(im, ax=ax, label="dB")

    ax2 = axes[1]
    ax2.plot(tf, prof_plot, lw=1.4, color="tab:blue", label="smoothed similarity")
    ax2.axhline(med, color="k", lw=0.8, alpha=0.5)
    ax2.axhline(thr, color="tab:red", lw=1.2, ls="--", label="threshold")
    ax2.axvspan(start_s, end_s, color="tab:red", alpha=0.12)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("mean cosine similarity\nto non-local frames")
    ax2.set_ylim(min(np.nanmin(prof_plot), thr) - 0.02, max(np.nanmax(prof_plot), 0.25))
    ax2.legend(loc="lower right", framealpha=0.8)
    ax2.set_title("MFCC self-similarity profile: the spliced phrase is the deep outlier")

    fig.tight_layout()
    out_png = HERE / "task_ethiopia_analysis.png"
    fig.savefig(out_png, dpi=110)
    print(f"  figure   : {out_png}")


if __name__ == "__main__":
    main()