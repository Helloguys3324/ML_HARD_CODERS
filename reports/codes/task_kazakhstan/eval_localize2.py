import cv2, numpy as np, glob, warnings, time
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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

def loo_localize():
    dists = []
    for holdout in range(len(names)):
        Xt, yt = [], []
        for gi in range(len(names)):
            if gi == holdout:
                continue
            img = cv2.imread(names[gi])
            mask = cv2.imread(names[gi].replace('.jpg', '_mask.png'), 0)
            F = np.vstack([ALL[gi][k].ravel() for k in KEY]).T
            m = (mask > 128).ravel()
            pos = F[m]
            neg_idx = np.where(~m)[0]
            rng = np.random.RandomState(0)
            n = min(len(pos) * 2, len(neg_idx))
            neg_idx = rng.choice(neg_idx, size=n, replace=False)
            Xt.append(np.vstack([pos, F[neg_idx]]))
            yt.append(np.concatenate([np.ones(len(pos)), np.zeros(n)]))
        Xt = np.vstack(Xt); yt = np.concatenate(yt)
        sc = StandardScaler().fit(Xt)
        clf = LogisticRegression(max_iter=500, class_weight='balanced').fit(sc.transform(Xt), yt)
        mask = cv2.imread(names[holdout].replace('.jpg', '_mask.png'), 0)
        F = np.vstack([ALL[holdout][k].ravel() for k in KEY]).T
        prob = clf.predict_proba(sc.transform(F))[:, 1].reshape(mask.shape)
        ys, xs = np.where(mask > 128); cy, cx = ys.mean(), xs.mean()
        # argmax after smoothing at multiple scales, pick best
        best = 1e9
        for s in (10, 20, 30, 40, 60):
            sm = cv2.GaussianBlur(prob.astype(np.float32), (0, 0), s)
            ay, ax = np.unravel_index(np.argmax(sm), sm.shape)
            best = min(best, np.hypot(ay - cy, ax - cx))
        dists.append(best)
    return np.array(dists)

d = loo_localize()
print(f'LOO localization (LogReg): median {np.median(d):.0f}px  mean {d.mean():.0f}px  <=25px {(d<=25).mean():.2%}  <=50px {(d<=50).mean():.2%}')