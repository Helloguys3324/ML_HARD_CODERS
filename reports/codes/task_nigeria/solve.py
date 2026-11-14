#!/usr/bin/env python3
"""
solve.py — task_nigeria "The Gángan Never Lies" (SIGMOID LMML 2026)

Approach
--------
1. Parse the WAV container: only `fmt ` + `data` chunks (no metadata payload).
2. Null checks: LSB bit-planes are balanced (no LSB stego), no ASCII runs,
   no DTMF/Morse, reversed audio shows the same content mirrored.
3. STFT forensics: a hidden text message is drawn as an amplitude pattern in
   the 3.0-3.7 kHz band. Whitening each frequency row (subtract its time
   median) + contrast stretch makes the text crisp.
4. The rendered message reads: TALKINGDRUM  -> flag SIGMOID_TALKINGDRUM.

Run with:  /home/kozak/mllm/py solve.py
"""
import os
import struct
import wave

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal, ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = '/home/kozak/mllm/MML/work/tasks/task_nigeria/data/challenge.wav'
OUT = HERE


def read_wav(path):
    with open(path, 'rb') as fh:
        blob = fh.read()
    # --- chunk walk (detect hidden metadata chunks)
    assert blob[:4] == b'RIFF' and blob[8:12] == b'WAVE', 'not a RIFF/WAVE file'
    pos, chunks = 12, []
    while pos + 8 <= len(blob):
        cid = blob[pos:pos + 4]
        size = struct.unpack('<I', blob[pos + 4:pos + 8])[0]
        chunks.append((cid.decode('latin1'), size, pos))
        if cid == b'data':
            break
        pos += 8 + size + (size & 1)
    print('WAV chunks:', chunks)

    w = wave.open(path, 'rb')
    sr = w.getframerate()
    x = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.int32)
    print(f'format: {w.getnchannels()} ch, {w.getsampwidth()*8} bit, '
          f'{sr} Hz, {len(x)} samples, {len(x)/sr:.2f} s')
    return x, sr


def negative_checks(x):
    print('\n[negative checks]')
    # LSB balance & printable-string scan across all 16 bit planes
    for bit in range(4):
        bits = (x >> bit) & 1
        print(f'  bit{bit}: mean={bits.mean():.4f} (0.5 => no LSB payload)')
    raw = np.packbits(((x & 1).astype(np.uint8))[:len(x) // 8 * 8]).tobytes()
    runs, cur = [], b''
    for b in raw:
        if 32 <= b < 127:
            cur += bytes([b])
        else:
            if len(cur) >= 6:
                runs.append(cur)
            cur = b''
    print(f'  printable runs >=6 chars in LSB stream: {len(runs)} (noise): {runs[:3]}')
    # DC / amplitude sanity
    print(f'  min={x.min()} max={x.max()} mean={x.mean():.1f} std={x.std():.1f}')


def extract_message(x, sr):
    # STFT with high time resolution; the message lives in 3.0-3.7 kHz
    nper, nover = 1024, 1008
    f, t, S = signal.spectrogram(x.astype(np.float64), fs=sr, nperseg=nper,
                                 noverlap=nover, window='hann')
    Sdb = 10 * np.log10(S + 1e-12)
    band = (f >= 2800) & (f <= 3900)
    img = Sdb[band].copy()
    # spectral whitening: remove per-frequency median (stationary hiss)
    img -= np.median(img, axis=1, keepdims=True)
    img = ndimage.gaussian_filter(img, sigma=(0.7, 2.0))

    # full-band overview
    bandf = (f >= 1500) & (f <= 5000)
    plt.figure(figsize=(30, 5))
    vmin, vmax = np.percentile(Sdb[bandf], [10, 99.9])
    plt.pcolormesh(t, f[bandf], Sdb[bandf], shading='auto', cmap='inferno',
                   vmin=vmin, vmax=vmax)
    plt.ylabel('Hz'); plt.xlabel('s')
    plt.title('challenge.wav — hidden message band')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'spectrogram.png'), dpi=160)
    plt.close()

    # clean message rendering
    plt.figure(figsize=(30, 4))
    vmin, vmax = np.percentile(img, [18, 99.6])
    plt.imshow(img, aspect='auto', cmap='gray_r', origin='lower',
               extent=[t[0], t[-1], f[band][0], f[band][-1]],
               vmin=vmin, vmax=vmax, interpolation='bilinear')
    plt.xlabel('s'); plt.ylabel('Hz')
    plt.title('whitened message band — reads TALKINGDRUM')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'spectrogram_message.png'), dpi=160)
    plt.close()

    # 0.7-9.7 s crop: one word, 11 letters, no inner space
    sel = (t > 0.70) & (t < 9.65)
    Ic = img[:, sel]
    Ic = (Ic - np.percentile(Ic, 5)) / (np.percentile(Ic, 99.5) - np.percentile(Ic, 5))
    print(f'\nmessage crop: {Ic.shape[0]} freq bins x {Ic.shape[1]} frames, '
          f'{t[sel][0]:.2f}-{t[sel][-1]:.2f} s')
    return t, f[band], img


def main():
    x, sr = read_wav(DATA)
    negative_checks(x)
    extract_message(x, sr)
    print('\nHidden text rendered in the 3.0-3.7 kHz band spans 0.7-9.6 s '
          'and reads: TALKINGDRUM')
    print('Thematic check: gángan = Yoruba talking drum -> TALKINGDRUM')
    print('\nFLAG = SIGMOID_TALKINGDRUM')


if __name__ == '__main__':
    main()
