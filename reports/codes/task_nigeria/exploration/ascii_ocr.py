import wave, numpy as np, os
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
from scipy import signal, ndimage

DATA = '/home/kozak/mllm/MML/work/tasks/task_nigeria/data/challenge.wav'
sr = 22050
w = wave.open(DATA, 'rb')
x = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float64)
x -= x.mean()

# --- high time resolution STFT in the message band
nper, nover = 1024, 1008
f, t, S = signal.spectrogram(x, fs=sr, nperseg=nper, noverlap=nover, window='hann')
Sdb = 10*np.log10(S + 1e-12)
band = (f >= 2900) & (f <= 3800)
img = Sdb[band].copy()
img -= np.median(img, axis=1, keepdims=True)
img = ndimage.gaussian_filter(img, sigma=(0.7, 2.5))
print('img', img.shape, 'dt', t[1]-t[0])

# keep message span
sel = (t > 0.5) & (t < 9.9)
tt = t[sel]
I = img[:, sel]
# binary ink = positive deviation, weighted by frequency consistency:
ink = (I > 1.5).astype(float)
prof = ink.mean(axis=0)
# smooth profile
prof = ndimage.gaussian_filter1d(prof, sigma=6)
on = prof > 0.18
# merge gaps < 40 ms
idx = np.flatnonzero(np.diff(on.astype(int)) != 0)
segs = []
start = 0 if on[0] else None
for k in range(len(idx)):
    pos = idx[k]
    if on[pos] and start is None:
        start = pos
    elif not on[pos] and start is not None:
        segs.append((start, pos)); start = None
if start is not None: segs.append((start, len(on)))
segs = [(a, b) for a, b in segs if (b-a)*(t[1]-t[0]) > 0.2]
# merge close segments
merged = []
for a, b in segs:
    if merged and (a - merged[-1][1])*(t[1]-t[0]) < 0.09:
        merged[-1] = (merged[-1][0], b)
    else:
        merged.append((a, b))
print('segments:', len(merged))
for a, b in merged:
    print('  %.2f-%.2f dur %.2f' % (tt[a], tt[b], tt[b]-tt[a]))

# print ASCII art of each segment (downsample to 24 wide x 10 tall)
for si, (a, b) in enumerate(merged):
    cell = ink[:, a:b]
    if cell.shape[1] < 8: continue
    im = Image.fromarray((cell*255).astype(np.uint8)).resize((16, 9), Image.LANCZOS)
    arr = np.array(im)
    chars = ' .:-=+*#%@'
    print(f'--- seg {si} t={tt[a]:.2f}-{tt[b]:.2f}')
    for row in arr:
        print('   ' + ''.join(chars[min(9, int(v/25.6))] for v in row))
