"""Attempt automated glyph segmentation + template OCR of the spectrogram text."""
import wave, numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from scipy import signal, ndimage

DATA = '/home/kozak/mllm/MML/work/tasks/task_nigeria/data/challenge.wav'
OUT = '/home/kozak/mllm/MML/work/solutions/task_nigeria'

w = wave.open(DATA, 'rb')
sr = w.getframerate()
x = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float64)
x -= x.mean()

# STFT fine time resolution
nper = 512
nover = 960 if 960 < nper else nper-16   # noverlap must be < nperseg
nover = 496
f, t, S = signal.spectrogram(x, fs=sr, nperseg=nper, noverlap=nover, window='hann')
Sdb = 10*np.log10(S + 1e-12)
band = (f >= 2900) & (f <= 3800)
img = Sdb[band]
# whiten: subtract per-frequency median over time (stationary noise removal)
med = np.median(img, axis=1, keepdims=True)
imgw = img - med
# smooth in time a bit
imgw = ndimage.gaussian_filter(imgw, sigma=(1.0, 2.0))
thr = np.percentile(imgw, 88)
mask = imgw > thr
# clean up
mask = ndimage.binary_closing(mask, structure=np.ones((2,2)))
mask = ndimage.binary_opening(mask, structure=np.ones((2,2)))
# fill holes
mask = ndimage.binary_fill_holes(mask)
lab, n = ndimage.label(mask)
sizes = ndimage.sum(mask, lab, range(1, n+1))
print('components:', n, 'largest areas:', sorted(sizes)[-25:])

# inspect component bounding boxes with area > 150
objs = []
for i in range(1, n+1):
    if sizes[i-1] < 150: continue
    ys, xs = np.where(lab == i)
    objs.append((xs.min(), xs.max(), ys.min(), ys.max(), sizes[i-1]))
objs.sort()
print('kept', len(objs))
for o in objs:
    print('  x[%4d-%4d] y[%2d-%2d] area %5d  t=%.2f-%.2f' % (o[0], o[1], o[2], o[3], o[4], t[o[0]], t[o[1]]))

# save mask visualization
plt.figure(figsize=(28, 4))
plt.imshow(mask, aspect='auto', cmap='gray_r', origin='lower',
           extent=[t[0], t[-1], f[band][0], f[band][-1]])
plt.title('segmentation mask'); plt.xlabel('s'); plt.ylabel('Hz')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'mask.png'), dpi=150)
