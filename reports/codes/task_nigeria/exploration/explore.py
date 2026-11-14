import wave, numpy as np, os, sys

DATA = '/home/kozak/mllm/MML/work/tasks/task_nigeria/data/challenge.wav'
OUT = '/home/kozak/mllm/MML/work/solutions/task_nigeria'

w = wave.open(DATA, 'rb')
sr = w.getframerate()
x = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.int32)
print('samples', len(x), 'sr', sr, 'dur', len(x)/sr)

# ---- 1. bit planes -> ASCII scan
def scan_bits(bits, label):
    # pack MSB-first into bytes
    n = len(bits)//8*8
    b = np.packbits(bits[:n].astype(np.uint8))
    raw = b.tobytes()
    # printable runs
    runs = []
    cur = b''
    for byte in raw:
        if 32 <= byte < 127 or byte in (9,10,13):
            cur += bytes([byte])
        else:
            if len(cur) >= 4:
                runs.append(cur)
            cur = b''
    if len(cur) >= 4: runs.append(cur)
    return runs

for bit in range(16):
    bits = (x >> bit) & 1
    runs = scan_bits(bits, f'bit{bit}')
    good = [r for r in runs if b'SIGMOID' in r or b'flag' in r.lower() or b'FLAG' in r]
    print(f'bit {bit}: {len(runs)} printable runs >=4')
    if runs:
        print('   sample:', runs[:5])
    if good:
        print('   *** FLAG-LIKE:', good)

# ---- 2. LSB of absolute / signed magnitude etc
for name, arr in [('signed', x), ('abs', np.abs(x))]:
    for bit in range(4):
        bits = (arr >> bit) & 1
        runs = scan_bits(bits, name)
        if runs:
            print(f'{name} bit{bit} runs:', runs[:3])

# ---- 3. simple spectrogram + save arrays
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal

f, t, Sxx = signal.spectrogram(x.astype(np.float64), fs=sr, nperseg=1024, noverlap=768)
Sdb = 10*np.log10(Sxx + 1e-12)
plt.figure(figsize=(16, 8))
plt.pcolormesh(t, f, Sdb, shading='auto', cmap='inferno')
plt.colorbar(); plt.ylabel('Hz'); plt.xlabel('s'); plt.title('challenge.wav spectrogram')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'spectrogram.png'), dpi=120)
print('saved spectrogram')

# ---- 4. spectral stats: dominant frequencies
frame = 2048
hop = 1024
print('\nFrame dominant freqs (first 30 frames):')
for i in range(0, min(len(x)-frame, hop*30), hop):
    seg = x[i:i+frame].astype(np.float64)
    seg = seg - seg.mean()
    spec = np.abs(np.fft.rfft(seg*np.hanning(frame)))
    freqs = np.fft.rfftfreq(frame, 1/sr)
    top = np.argsort(spec)[-4:][::-1]
    print(f'  t={i/sr:6.3f}s peak={freqs[top[0]]:8.1f}Hz  others={[round(float(freqs[j]),1) for j in top[1:]]}')

# ---- 5. whole-file average spectrum peaks
seg = x.astype(np.float64) - x.mean()
spec = np.abs(np.fft.rfft(seg))
freqs = np.fft.rfftfreq(len(seg), 1/sr)
top = np.argsort(spec)[-20:][::-1]
print('\nGlobal top-20 spectral peaks:')
for j in top:
    print(f'  {freqs[j]:8.1f} Hz  mag {spec[j]:.3e}')

# ---- 6. check tones: find strong narrowband components
print('\nRatio of energy above 4kHz:', np.sum(spec[freqs>4000]**2)/np.sum(spec**2))
