import cv2, numpy as np, glob, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from learn_detector import extract

names = sorted(glob.glob('challenges/task_kazakhstan/data/exhibits/exhibit_*.jpg'))
KEY = None
# precompute feature maps for all exhibits
ALL = {}
for gi, p in enumerate(names):
    img = cv2.imread(p)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    maps = extract(g, img)
    if KEY is None:
        KEY = list(maps.keys())
    ALL[gi] = maps

dists = []
for holdout in range(len(names)):
    # build train
    Xt, yt = [], []
    for gi, p in enumerate(names):
        if gi == holdout:
            continue
        img = cv2.imread(p)
        mask = cv2.imread(p.replace('.jpg', '_mask.png'), 0)
        maps = ALL[gi]
        F = np.vstack([maps[k].ravel() for k in KEY]).T
        m = (mask > 128).ravel()
        pos = F[m]
        neg_idx = np.where(~m)[0]
        rng = np.random.RandomState(holdout)
        neg_idx = rng.choice(neg_idx, size=min(len(pos) * 2, len(neg_idx)), replace=False)
        Xt.append(np.vstack([pos, F[neg_idx]]))
        yt.append(np.concatenate([np.ones(len(pos)), np.zeros(len(neg_idx))]))
    Xt = np.vstack(Xt); yt = np.concatenate(yt)
    scaler = StandardScaler().fit(Xt)
    clf = LogisticRegression(max_iter=800, class_weight='balanced').fit(scaler.transform(Xt), yt)
    # predict held-out
    img = cv2.imread(names[holdout])
    mask = cv2.imread(names[holdout].replace('.jpg', '_mask.png'), 0)
    maps = ALL[holdout]
    F = np.vstack([maps[k].ravel() for k in KEY]).T
    prob = clf.predict_proba(scaler.transform(F))[:, 1].reshape(mask.shape)
    prob = cv2.GaussianBlur(prob.astype(np.float32), (0, 0), 25)
    ay, ax = np.unravel_index(np.argmax(prob), prob.shape)
    ys, xs = np.where(mask > 128)
    cy, cx = ys.mean(), xs.mean()
    dists.append(np.hypot(ay - cy, ax - cx))
    print(f'{names[holdout].split("/")[-1]}: pred=({ax},{ay}) mask=({cx:.0f},{cy:.0f}) dist={dists[-1]:.0f}')

d = np.array(dists)
print('---')
print(f'localization: median dist {np.median(d):.0f}px, mean {d.mean():.0f}px, <=30px: {(d<=30).mean():.2%}, <=50px: {(d<=50).mean():.2%}')