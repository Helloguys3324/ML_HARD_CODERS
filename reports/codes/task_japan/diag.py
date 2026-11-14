import numpy as np, glob, os
from PIL import Image
from scipy import ndimage

ROOT = '/home/kozak/mllm/MML/work/tasks/task_japan/data/dataset'
OUT = '/home/kozak/mllm/MML/work/solutions/task_japan'

def res(path, sigma=1.2):
    g = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32).mean(axis=2)
    r = g - ndimage.gaussian_filter(g, sigma)
    return r[40:850, 40:1240]

d = np.load(os.path.join(OUT, 'fps.npz'))
fps = {k: d[k] for k in d.files}

# 1. average of anonymous residuals -> any common structure?
anon = sorted(glob.glob(os.path.join(ROOT, 'anonymous', '*.jpg')))
acc = None
for p in anon[::5]:
    r = res(p)
    acc = r if acc is None else acc + r
avg = acc / len(anon[::5])
print('avg anon residual std', avg.std(), 'per-image resid std approx 13 -> random mean std would be ~13/sqrt(29)=2.4')
print('avg anon resid max abs', np.abs(avg).max())

# correlation of average anon with each fingerprint
def ncc(x, y):
    x = x.ravel()-x.mean(); y=y.ravel()-y.mean()
    return float((x*y).mean()/(x.std()*y.std()+1e-12))
print('avg-anon vs fingerprints:', {h: round(ncc(avg, fps[h]),4) for h in fps})

# 2. average residual within credited person vs within credited: known good
# check if credited avg correlates with each individual credited image of same person (sanity)
for h in ['toorisugari']:
    ps = sorted(glob.glob(os.path.join(ROOT,'credited',h,'*.jpg')))
    acc=None
    for p in ps:
        r=res(p); acc = r if acc is None else acc+r
    a=acc/len(ps)
    print(h, 'self corr with images:', [round(ncc(a,res(p)),3) for p in ps[:6]])

# 3. shift search: anonymous image vs toorisugari fingerprint
p = anon[0]
r = res(p)
best = (-9, None)
for dy in range(-8, 9, 2):
    for dx in range(-8, 9, 2):
        rr = r[max(0,dy):r.shape[0]+min(0,dy), max(0,dx):r.shape[1]+min(0,dx)]
        ff = fps['toorisugari'][max(0,-dy):fps['toorisugari'].shape[0]+min(0,-dy), max(0,-dx):fps['toorisugari'].shape[1]+min(0,-dx)]
        v = ncc(rr, ff[:rr.shape[0],:rr.shape[1]])
        if v > best[0]: best = (v, (dy,dx))
print('best shift corr anon[0] vs toorisugari:', best)

# 4. compare anonymous pairwise: first 20 vs last 20 average
def avgset(ps):
    acc=None
    for p in ps:
        r=res(p); acc = r if acc is None else acc+r
    return acc/len(ps)
a1 = avgset(anon[:20]); a2 = avgset(anon[-20:])
print('anon first20-avg vs last20-avg ncc', ncc(a1,a2))
print('anon first20-avg std', a1.std())
