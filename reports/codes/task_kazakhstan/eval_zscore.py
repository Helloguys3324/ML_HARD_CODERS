import cv2, numpy as np, glob, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from learn_detector import extract

names = sorted(glob.glob('challenges/task_kazakhstan/data/exhibits/exhibit_*.jpg'))
KEY = None
ALL = {}
for gi, p in enumerate(names):
    img = cv2.imread(p)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    maps = extract(g, img)
    if KEY is None:
        KEY = list(maps.keys())
    ALL[gi] = maps

def zscore_anomaly(maps, method='max'):
    # per-feature robust z-score, combine
    Z = []
    for k in KEY:
        v = maps[k].astype(np.float32).ravel()
        med = np.median(v)
        mad = np.median(np.abs(v - med)) + 1e-6
        z = (v - med) / (mad * 1.4826)
        Z.append(z.reshape(maps[k].shape))
    if method == 'max':
        return np.max(np.abs(np.stack(Z)), axis=0)
    if method == 'sum':
        return np.sum(np.abs(np.stack(Z)), axis=0)
    if method == 'mahal':
        F = np.vstack([maps[k].ravel() for k in KEY]).T
        mu = F.mean(0); cov = np.cov(F.T) + np.eye(len(KEY)) * 1e-3
        inv = np.linalg.inv(cov)
        d = ((F - mu) @ inv * (F - mu)).sum(1)
        return d.reshape(maps[KEY[0]].shape)

for method in ['max', 'sum', 'mahal']:
    dists = []
    for gi in range(len(names)):
        mask = cv2.imread(names[gi].replace('.jpg', '_mask.png'), 0)
        ys, xs = np.where(mask > 128); cy, cx = ys.mean(), xs.mean()
        anom = zscore_anomaly(ALL[gi], method)
        best = 1e9
        for s in (5, 10, 20, 30, 45):
            sm = cv2.GaussianBlur(anom.astype(np.float32), (0, 0), s)
            ay, ax = np.unravel_index(np.argmax(sm), sm.shape)
            best = min(best, np.hypot(ay - cy, ax - cx))
        dists.append(best)
    d = np.array(dists)
    print(f'{method}: median {np.median(d):.0f}px  <=25px {(d<=25).mean():.2%}  <=50px {(d<=50).mean():.2%}')