"""Objective check: correlate extracted spectrogram text image against rendered
candidate words (DejaVu fonts) to confirm the spelling."""
import wave, numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
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
img = img - np.median(img, axis=1, keepdims=True)      # whiten stationary noise
img = ndimage.gaussian_filter(img, sigma=(0.8, 1.5))
# crop message time range
tm = (t >= 0.55) & (t <= 9.85)
G = img[:, tm]
G = (G - np.percentile(G, 5)) / (np.percentile(G, 99.5) - np.percentile(G, 5) + 1e-9)
G = np.clip(G, 0, 1)
print('message image shape', G.shape, 'aspect w/h', G.shape[1]/G.shape[0])
np.save(os.path.join(OUT, 'message_matrix.npy'), G)

plt.figure(figsize=(28, 3))
plt.imshow(G, aspect='auto', cmap='gray_r', origin='lower',
           extent=[t[tm][0], t[tm][-1], f[band][0], f[band][-1]])
plt.xlabel('s'); plt.ylabel('Hz'); plt.title('extracted message (whitened)')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'message_extracted.png'), dpi=150)
plt.close()

# render candidate words
font_path = font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans', weight='bold'))
print('font', font_path)
H, W = G.shape
rendered = {}
for word in ['TALKINGDRUM', 'TALKINGDRUMS', 'TALKING DRUM', 'TALKING', 'DRUM',
             'GANGAN', 'IBADAN', 'YORUBA', 'AYAN', 'TALKINGDRUMMER', 'SIGMOID']:
    img_font = ImageFont.truetype(font_path, 200)
    tmp = Image.new('L', (4000, 400), 0)
    d = ImageDraw.Draw(tmp)
    d.text((20, 20), word, fill=255, font=img_font)
    arr = np.array(tmp)
    ys, xs = np.where(arr > 30)
    if len(ys) == 0: continue
    crop = arr[ys.min():ys.max()+1, xs.min():xs.max()+1]
    im = Image.fromarray(crop).resize((W, H), Image.LANCZOS)
    r = np.array(im, dtype=np.float64)/255.0
    r = ndimage.gaussian_filter(r, 0.7)
    # normalized cross correlation
    a = (G - G.mean())/G.std()
    b = (r - r.mean())/(r.std()+1e-9)
    ncc = float(np.mean(a*b))
    rendered[word] = ncc

for k, v in sorted(rendered.items(), key=lambda kv: -kv[1]):
    print(f'{k!r:20s} NCC={v:.4f}')
