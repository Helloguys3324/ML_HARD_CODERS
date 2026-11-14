import wave, numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal

DATA = '/home/kozak/mllm/MML/work/tasks/task_nigeria/data/challenge.wav'
OUT = '/home/kozak/mllm/MML/work/solutions/task_nigeria'

w = wave.open(DATA, 'rb')
sr = w.getframerate()
x = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.int64)

# --- LSB bitmap test over several widths
b0 = (x & 1).astype(np.uint8)
for width in (512, 586, 672, 1024):
    h = len(b0)//width
    if h < 10: continue
    img = b0[:h*width].reshape(h, width)
    plt.figure(figsize=(8, 8*h/width))
    plt.imshow(img, cmap='gray', aspect='auto', interpolation='nearest')
    plt.title(f'LSB bit0 width={width}'); plt.axis('off')
    plt.tight_layout(); plt.savefig(os.path.join(OUT, f'lsb_w{width}.png'), dpi=80)
    plt.close()

# --- reversed audio spectrogram
xr = x[::-1].astype(np.float64); xr -= xr.mean()
f, t, S = signal.spectrogram(xr, fs=sr, nperseg=1024, noverlap=768)
plt.figure(figsize=(16, 6))
plt.pcolormesh(t, f, 10*np.log10(S+1e-12), shading='auto', cmap='inferno')
plt.ylabel('Hz'); plt.xlabel('s'); plt.title('REVERSED audio')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'spectrogram_reversed.png'), dpi=120)
plt.close()

# --- per-bit statistical sanity: LSB balance (stego often biased)
for b in range(16):
    v = ((x >> b) & 1).mean()
    print(f'bit{b:2d} mean {v:.4f}')

# --- energy per 0.25s window to see any on/off keying outside the message band
from scipy import signal as sg
seg = 2205
for i in range(0, len(x)-seg, seg*2):
    rms = np.sqrt(np.mean((x[i:i+seg].astype(np.float64))**2))
    print(f'{i/sr:5.2f}s rms {rms:8.1f}')
