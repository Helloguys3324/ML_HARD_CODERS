import cv2, numpy as np, glob
from pathlib import Path

def texture_map(g, k=3):
    """local high-frequency energy (gradient magnitude), box-smoothed."""
    g = g.astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    tex = np.abs(lap)
    tex = cv2.boxFilter(tex, -1, (7, 7))
    return tex

def smooth_anomaly(g, inner=9, outer=25):
    """score = (median texture of outer annulus) - (texture at center),
    large positive => smooth 'hole' in textured area."""
    tex = texture_map(g)
    inner_m = cv2.boxFilter(tex, -1, (inner, inner))
    outer_m = cv2.boxFilter(tex, -1, (outer, outer))
    # annulus approx = outer minus inner (rough)
    anom = outer_m - inner_m
    return anom

def evaluate(feat_fn):
    hits = []
    ious = []
    for p in sorted(glob.glob('challenges/task_kazakhstan/data/exhibits/exhibit_*.jpg')):
        img = cv2.imread(p)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = cv2.imread(p.replace('.jpg', '_mask.png'), 0)
        ys, xs = np.where(mask > 128)
        cy, cx = ys.mean(), xs.mean()
        sig = feat_fn(g)
        sig = cv2.GaussianBlur(sig.astype(np.float32), (0, 0), 20)
        ay, ax = np.unravel_index(np.argmax(sig), sig.shape)
        dist = np.hypot(ay - cy, ax - cx)
        hits.append(dist < 30)
        # iou: threshold top region
        th = sig.max() * 0.6
        pred = sig > th
        inter = np.logical_and(pred, mask > 128).sum()
        union = np.logical_or(pred, mask > 128).sum()
        ious.append(inter / max(1, union))
    return np.mean(hits), np.mean(ious)

print('smooth_anomaly (removed detection):')
for inner, outer in [(9, 25), (9, 31), (7, 21)]:
    h, i = evaluate(lambda g, i=inner, o=outer: smooth_anomaly(g, i, o))
    print(f'  inner={inner} outer={outer}: hit>30px={h:.2f}, meanIoU={i:.3f}')