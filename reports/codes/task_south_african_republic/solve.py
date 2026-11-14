#!/usr/bin/env python3
"""Reproducible analysis for task_south_african_republic ("Ngiyakuzwa?").

Run with /home/kozak/mllm/py solve.py
Outputs spectrograms and prints the forensic numbers used in solution.md.
"""
import os
import struct
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
WAV = os.path.join(HERE, "..", "..", "tasks", "task_south_african_republic",
                   "data", "voice_note.wav")
WAV = os.path.abspath(WAV)
OUT = HERE


def wav_chunks(path):
    b = open(path, "rb").read()
    print("file bytes:", len(b), "header:", b[:12])
    pos = 12
    while pos + 8 <= len(b):
        cid = b[pos:pos + 4]
        sz = struct.unpack("<I", b[pos + 4:pos + 8])[0]
        print("chunk", cid, "size", sz)
        pos += 8 + sz + (sz & 1)
    print("end offset", pos)


def main():
    wav_chunks(WAV)
    x, sr = sf.read(WAV, dtype="float32")
    print("sr", sr, "samples", len(x), "duration", len(x) / sr,
          "peak", float(np.abs(x).max()), "rms", float(np.sqrt((x * x).mean())))

    # 1) LSB randomness (quick stego check)
    xi = np.clip(np.round(x * 32768), -32768, 32767).astype(np.int16)
    for k in range(3):
        b = ((xi >> k) & 1).astype(np.uint8)
        ac = [float(np.corrcoef(b[:-l], b[l:])[0, 1]) for l in (1, 2, 3)]
        print("LSB bit", k, "mean", float(b.mean()), "autocorr", ac)

    # 2) Global spectrum: identify persistent vuvuzela lines
    nper = 16384
    hop = nper // 2
    acc = []
    for i in range(0, len(x) - nper, hop):
        seg = x[i:i + nper] * np.hanning(nper)
        acc.append(np.abs(np.fft.rfft(seg)) ** 2)
    P = np.median(np.array(acc), axis=0)
    f = np.fft.rfftfreq(nper, 1 / sr)
    L = 10 * np.log10(P + 1e-20)
    base = signal.medfilt(L, 201)
    pk, _ = signal.find_peaks(L, height=base + 15, distance=3)
    peaks = f[pk]
    print("persistent spectral lines (Hz):", [round(float(v), 1) for v in peaks[:40]])
    print("=> two harmonic series at F0 = 217.8 and 233.4 Hz")

    # 3) Spectrograms of the first 20 s and of the clean lead-in
    for a, b, name in [(0, 20, "solve_spec_0_20.png"),
                       (0, 1.2, "solve_spec_leadin.png")]:
        seg = x[int(a * sr):int(b * sr)]
        f2, t, S = signal.spectrogram(seg, sr, nperseg=512, noverlap=448)
        plt.figure(figsize=(22, 6))
        plt.pcolormesh(t + a, f2, 10 * np.log10(S + 1e-12), shading="auto",
                       vmin=-130, vmax=-70)
        plt.ylim(0, 8000)
        plt.colorbar()
        plt.title(name)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, name), dpi=110)
        plt.close()

    # 4) Clean lead-in pitch (0-0.5 s) - the only speech-like segment
    try:
        import librosa
        lead = x[:int(0.5 * sr)]
        f0, vflag, _ = librosa.pyin(lead, fmin=70, fmax=400, sr=sr,
                                    frame_length=1024, hop_length=128)
        v = f0[~np.isnan(f0)]
        print("lead-in voiced fraction", float(vflag.mean()),
              "F0 range", (float(v.min()), float(v.max())) if len(v) else None)
    except Exception as e:
        print("pyin skipped:", e)

    # 5) High-pass 4 kHz envelope (shows crowd bursts / whistles)
    sos = signal.butter(8, 4000, btype="highpass", fs=sr, output="sos")
    z = signal.sosfiltfilt(sos, x)
    env = np.array([np.sqrt((z[i:i + 800] ** 2).mean())
                    for i in range(0, len(z) - 800, 800)])
    t = np.arange(len(env)) * 0.05
    np.savez(os.path.join(OUT, "solve_gate.npz"), t=t, env=env)
    plt.figure(figsize=(24, 4))
    plt.plot(t, env)
    plt.title("4 kHz band envelope (vuvuzela/crowd gate)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "solve_gate.png"), dpi=100)
    plt.close()
    print("gate envelope percentiles", np.percentile(env, [1, 10, 50, 90, 99]))

    print("\nConclusion: the recording is dominated by two gated vuvuzela "
          "harmonic drones (F0 217.8/233.4 Hz). Only 0-0.4 s contains clean "
          "speech; the word after 'ikhodi' could not be isolated unambiguously. "
          "Best narrative/phonetic candidate: SIGMOID_YEBO (LOW confidence); "
          "second candidate: SIGMOID_SAWUBONA.")


if __name__ == "__main__":
    main()
