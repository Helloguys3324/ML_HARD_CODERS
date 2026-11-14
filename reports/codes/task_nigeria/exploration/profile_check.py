import wave, numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal, ndimage

DATA = '/home/kozak/mllm/MML/work/tasks/task_nigeria/data/challenge.wav'
OUT = '/home/kozak/mllm/MML/work/solutions/task_nigeria'

w = wave.open(DATA, 'rb')
sr = w.getframerate()
x = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float64)
x -= x.mean()

nper, nover = 512, 496
f, t, S = signal.spectrogram(x, fs=sr, nperseg=nper, noverlap=nover, window='hann')
Sdb = 10*np.log10(S + 1e-12)
band = (f >= 2950) & (f <= 3750)
img = Sdb[band]
img = img - np.median(img, axis=1, keepdims=True)
img = ndimage.gaussian_filter(img, sigma=(0.6, 2.0))

prof = img.mean(axis=0)
# smooth
prof_s = ndimage.gaussian_filter1d(prof, sigma=8)
prof_n = (prof_s - np.percentile(prof_s, 10)) / (np.percentile(prof_s, 99) - np.percentile(prof_s, 10))
plt.figure(figsize=(26, 3))
plt.plot(t, prof_n, lw=0.7)
plt.xlabel('s'); plt.ylabel('band energy (norm)'); plt.grid(alpha=.3)
plt.title('column energy profile, 3.0-3.7 kHz')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'column_profile.png'), dpi=150)

# find letter blocks: threshold + minimum gap
thr = 0.35
on = prof_n > thr
# restrict to message span
on[:np.searchsorted(t, 0.5)] = False
on[np.searchsorted(t, 9.9):] = False
lab, n = ndimage.label(on)
blocks = []
for i in range(1, n+1):
    idx = np.where(lab == i)[0]
    dur = t[idx[-1]] - t[idx[0]]
    if dur < 0.15:  # merge tiny noise blobs away
        continue
    blocks.append((t[idx[0]], t[idx[-1]]))
print('blocks found:', len(blocks))
for b in blocks:
    print('  %.3f - %.3f  (%.3f s)' % (b[0], b[1], b[1]-b[0]))
