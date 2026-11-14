import cv2, numpy as np, glob
from pathlib import Path

def features(g, img):
    f = {}
    g8 = g.astype(np.float32)
    # ELA at multiple qualities
    for q in (75, 85, 90, 95):
        enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])[1]
        again = cv2.imdecode(enc, 1).astype(np.float32)
        ela = np.abs(img.astype(np.float32) - again).mean(2)
        f[f'ela{q}'] = ela
    # noise (median residuals)
    for k in (3, 5, 7):
        hp = g8 - cv2.medianBlur(g, k).astype(np.float32)
        f[f'noise{k}'] = np.abs(hp)
    # laplacian
    f['lap'] = np.abs(cv2.Laplacian(g, cv2.CV_32F))
    # median-filter residue large
    f['med13'] = np.abs(g8 - cv2.medianBlur(g, 13).astype(np.float32))
    f['med21'] = np.abs(g8 - cv2.medianBlur(g, 21).astype(np.float32))
    # bilateral residue
    bil = cv2.bilateralFilter(g, 9, 75, 75).astype(np.float32)
    f['bilat'] = np.abs(g8 - bil)
    # sobel magnitude
    sx = cv2.Sobel(g, cv2.CV_32F, 1, 0); sy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    f['sob'] = np.hypot(sx, sy)
    return f

def ascii_of(img, mask, W=120, H=40):
    return ""

def evaluate_one(path):
    img = cv2.imread(path)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.imread(path.replace('.jpg', '_mask.png'), 0)
    ys, xs = np.where(mask > 128)
    cy, cx = ys.mean(), xs.mean()
    feats = features(g, img)
    # inside vs outside ratio for each feature
    res = {}
    mm = (mask > 128)
    for name, sig in feats.items():
        inmean = sig[mm].mean()
        outmean = sig[~mm].mean()
        ratio = inmean / (outmean + 1e-6)
        # argmax-based center distance
        sm = cv2.boxFilter(sig.astype(np.float32), -1, (31, 31))
        ay, ax = np.unravel_index(np.argmax(sm), sm.shape)
        dist_max = np.hypot(ay - cy, ax - cx)
        smin = cv2.boxFilter((-sig).astype(np.float32), -1, (31, 31))
        ay2, ax2 = np.unravel_index(np.argmax(smin), smin.shape)
        dist_min = np.hypot(ay2 - cy, ax2 - cx)
        res[name] = (round(ratio, 3), round(dist_max), round(dist_min))
    return res

allres = {}
paths = sorted(glob.glob('challenges/task_kazakhstan/data/exhibits/exhibit_*.jpg'))
for p in paths:
    allres[Path(p).stem] = evaluate_one(p)

# Summarize: for each feature, report median ratio, % success (dist_max or dist_min < 25)
names = None
for p in paths:
    r = allres[Path(p).stem]
    if names is None:
        names = list(r.keys())
    break

print(f"{'feature':8s} {'medRatioInOut':>13s} {'%maxHit':>7s} {'%minHit':>7s}  (hit = center within 25px of mask)")
for name in names:
    ratios = [allres[Path(p).stem][name][0] for p in paths]
    hitmax = sum(1 for p in paths if allres[Path(p).stem][name][1] < 25) / len(paths)
    hitmin = sum(1 for p in paths if allres[Path(p).stem][name][2] < 25) / len(paths)
    print(f"{name:8s} {np.median(ratios):13.3f} {hitmax*100:7.0f} {hitmin*100:7.0f}")