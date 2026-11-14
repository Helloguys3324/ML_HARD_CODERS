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
sel = (t > 0.70) & (t < 9.65)
I = img[:, sel]
I = (I - np.percentile(I, 5)) / (np.percentile(I, 99.5) - np.percentile(I, 5))
I = np.clip(I, 0, 1)
H, W = 60, 1200
G = np.array(Image.fromarray((I*255).astype(np.uint8)).resize((W, H), Image.LANCZOS), dtype=float)/255
Image.fromarray((G*255).astype(np.uint8)).save(os.path.join(OUT, 'extracted_word.png'))

def ncc(a, b):
    a = (a-a.mean())/a.std(); b = (b-b.mean())/b.std()
    return float((a*b).mean())

fonts = ['DejaVu Sans:bold', 'DejaVu Serif:bold', 'DejaVu Sans Mono:bold']
cands = ['TALKINGDRUM', 'TALKINGDRUMS', 'TALKING DRUM', 'TALKINGDRUMMER',
         'TALKING', 'DRUM', 'GANGAN', 'IBADAN', 'YORUBA', 'AYAN', 'LAGOS',
         'SIGMOID', 'TALKINGDRUN', 'TALKINGORUM', 'TALKINGDRAM', 'TALKINGDRUMX']
best = {}
for fs in fonts:
    fam, wt = fs.split(':')
    fp = font_manager.findfont(font_manager.FontProperties(family=fam, weight=wt))
    for c in cands:
        fnt = ImageFont.truetype(fp, 220)
        tmp = Image.new('L', (8000, 500), 0)
        d = ImageDraw.Draw(tmp)
        d.text((30, 60), c, fill=255, font=fnt)
        arr = np.array(tmp)
        ys, xs = np.where(arr > 20)
        crop = arr[ys.min():ys.max()+1, xs.min():xs.max()+1]
        im = Image.fromarray(crop).resize((W, H), Image.LANCZOS)
        r = np.array(im, dtype=float)/255
        r = ndimage.gaussian_filter(r, 0.8)
        best[(fs, c)] = ncc(G, r)
top = sorted(best.items(), key=lambda kv: -kv[1])[:12]
for (fs, c), s in top:
    print(f'{fs:22s} {c:15s} {s:+.4f}')
