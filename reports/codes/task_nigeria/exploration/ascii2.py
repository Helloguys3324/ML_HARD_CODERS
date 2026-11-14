import wave, numpy as np, os
from PIL import Image
from scipy import signal, ndimage

DATA = '/home/kozak/mllm/MML/work/tasks/task_nigeria/data/challenge.wav'
sr = 22050
w = wave.open(DATA, 'rb')
x = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float64)
x -= x.mean()

nper, nover = 1024, 1008
f, t, S = signal.spectrogram(x, fs=sr, nperseg=nper, noverlap=nover, window='hann')
Sdb = 10*np.log10(S + 1e-12)
band = (f >= 2900) & (f <= 3800)
img = Sdb[band].copy()
img -= np.median(img, axis=1, keepdims=True)
img = ndimage.gaussian_filter(img, sigma=(0.7, 2.5))
sel = (t > 0.5) & (t < 9.9)
tt = t[sel]; I = img[:, sel]
print('I stats', np.percentile(I, [1,10,50,90,99]))

# binarize relative to per-frame contrast: for each time col, count rows with strong value
strong = I > 2.0
prof = strong.mean(axis=0)
prof = ndimage.gaussian_filter1d(prof, sigma=4)
print('prof stats', np.percentile(prof, [1,25,50,75,90,99]))
on = prof > 0.10
print('on fraction', on.mean())
idx = np.flatnonzero(np.diff(on.astype(int)) != 0)
print('edges', [round(float(tt[i]),2) for i in idx][:80])
