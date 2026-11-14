import cv2, numpy as np, glob
from pathlib import Path

def extract(g, img):
    """Return list of feature maps (2D arrays) for a grayscale+color image."""
    g8 = g.astype(np.float32)
    maps = {}
    # gradients
    gx = cv2.Sobel(g8, cv2.CV_32F, 1, 0); gy = cv2.Sobel(g8, cv2.CV_32F, 0, 1)
    grad = np.hypot(gx, gy)
    maps['grad_mean'] = cv2.boxFilter(grad, -1, (9, 9))
    maps['grad_std'] = cv2.boxFilter(grad * grad, -1, (9, 9)) - maps['grad_mean'] ** 2
    lap = cv2.Laplacian(g8, cv2.CV_32F)
    maps['lap_abs'] = cv2.boxFilter(np.abs(lap), -1, (9, 9))
    # noise (median residual)
    for k, name in [(3, 'r3'), (5, 'r5'), (9, 'r9')]:
        res = g8 - cv2.medianBlur(g, k).astype(np.float32)
        maps['noise_' + name] = cv2.boxFilter(np.abs(res), -1, (9, 9))
    # median filter residue (large)
    maps['med13'] = cv2.boxFilter(np.abs(g8 - cv2.medianBlur(g, 13).astype(np.float32)), -1, (9, 9))
    # ELA
    for q in (80, 90, 95):
        enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])[1]
        again = cv2.imdecode(enc, 1).astype(np.float32)
        ela = np.abs(img.astype(np.float32) - again).mean(2)
        maps[f'ela{q}'] = cv2.boxFilter(ela, -1, (9, 9))
    # color std (BGR) local
    col = img.astype(np.float32)
    cs = cv2.boxFilter((col ** 2).sum(2), -1, (9, 9)) - cv2.boxFilter(col.sum(2), -1, (9, 9)) ** 2 / 9
    maps['color_std'] = cs
    # local mean/std of gray
    mu = cv2.boxFilter(g8, -1, (9, 9))
    maps['gray_std'] = cv2.boxFilter(g8 * g8, -1, (9, 9)) - mu * mu
    return maps

def build_dataset():
    X, y, groups = [], [], []
    names = list(sorted(glob.glob('challenges/task_kazakhstan/data/exhibits/exhibit_*.jpg')))
    key_order = None
    for gi, p in enumerate(names):
        img = cv2.imread(p)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = cv2.imread(p.replace('.jpg', '_mask.png'), 0)
        maps = extract(g, img)
        if key_order is None:
            key_order = list(maps.keys())
        feats = [maps[k].ravel() for k in key_order]
        F = np.vstack(feats).T  # (N, D)
        mask_flat = (mask > 128).ravel()
        # sample: all positive, equal number negative
        pos = F[mask_flat]
        neg_idx = np.where(~mask_flat)[0]
        rng = np.random.RandomState(0)
        neg_idx = rng.choice(neg_idx, size=min(len(pos) * 3, len(neg_idx)), replace=False)
        neg = F[neg_idx]
        X.append(np.vstack([pos, neg]))
        y.append(np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]))
        groups.append(np.full(len(pos) + len(neg), gi))
    X = np.vstack(X); y = np.concatenate(y); groups = np.concatenate(groups)
    return X, y, groups, key_order

if __name__ == '__main__':
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    X, y, groups, keys = build_dataset()
    print('features', keys)
    print('X', X.shape, 'pos frac', y.mean())
    # logistic regression with group CV
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight='balanced'))
    scores = cross_val_score(clf, X, y, groups=groups, cv=GroupKFold(5), scoring='roc_auc', n_jobs=4)
    print('LogReg GroupKFold ROC-AUC:', scores, 'mean', scores.mean())
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, n_jobs=8, class_weight='balanced')
    scores = cross_val_score(rf, X, y, groups=groups, cv=GroupKFold(5), scoring='roc_auc', n_jobs=2)
    print('RF GroupKFold ROC-AUC:', scores, 'mean', scores.mean())