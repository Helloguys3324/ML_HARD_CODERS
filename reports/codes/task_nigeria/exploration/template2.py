import wave, numpy as np, os
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager
from scipy import signal, ndimage

DATA = '/home/kozak/mllm/MML/work/tasks/task_nigeria/data/challenge.wav'
OUT = '/home/kozak/mllm/MML/work/solutions/task_nigeria'
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
sel = (t > 0.55) & (t < 9.9)
I = img[:, sel]
# normalize
I = (I - np.percentile(I, 5)) / (np.percentile(I, 99.5) - np.percentile(I, 5))
I = np.clip(I, 0, 1)
H, W = 60, 1500
G = np.array(Image.fromarray((I*255).astype(np.uint8)).resize((W, H), Image.LANCZOS), dtype=float)/255

font_path = font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans', weight='bold'))

def render(word, size=(W, H)):
    fnt = ImageFont.truetype(font_path, 220)
    tmp = Image.new('L', (6000, 500), 0)
    d = ImageDraw.Draw(tmp)
    d.text((30, 60), word, fill=255, font=fnt)
    arr = np.array(tmp)
    ys, xs = np.where(arr > 20)
    crop = arr[ys.min():ys.max()+1, xs.min():xs.max()+1]
    im = Image.fromarray(crop).resize(size, Image.LANCZOS)
    r = np.array(im, dtype=float)/255
    return ndimage.gaussian_filter(r, 0.8)

def ncc(a, b):
    a = (a-a.mean())/a.std(); b = (b-b.mean())/b.std()
    return float((a*b).mean())

cands = ['TALKINGDRUM', 'TALKINGDRUMS', 'TALKING DRUM', 'TALKINGDRUMMER',
         'TALKING', 'DRUM', 'GANGAN', 'IBADAN', 'YORUBA', 'AYAN', 'LAGOS',
         'SIGMOID', 'TALKINGDRUMX', 'TALKINGDRUN', 'TALKINGORUM', 'TALKINGDRAM']
res = sorted([(ncc(G, render(c)), c) for c in cands], reverse=True)
for s, c in res:
    print(f'{c:18s} NCC={s:.4f}')
