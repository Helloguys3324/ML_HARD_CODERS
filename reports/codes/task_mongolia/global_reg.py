#!/usr/bin/env python
"""Global image features + regression to sheep counts. Fast baseline."""
import os, sys, time, json
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import maximum_filter

DATA = '/home/kozak/mllm/MML/work/tasks/task_mongolia/data'
OUT = '/home/kozak/mllm/MML/work/solutions/task_mongolia'
cv2.setNumThreads(2)


def feats(img_bgr):
    img = img_bgr.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[:, :, 1] / 255.0
    v = hsv[:, :, 2] / 255.0
    white = v * (1.0 - s) ** 1.2
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] / 255.0
    f = []
    for t in (0.3, 0.4, 0.5, 0.6, 0.7):
        f.append(float((white > t).mean()))
    f += [float(white.mean()), float(white.std()), float(np.percentile(white, 90)),
          float(np.percentile(white, 99))]
    f += [float(v.mean()), float(s.mean()), float(L.mean()), float(L.std())]
    # top-hat areas
    w8 = np.clip(white * 255, 0, 255).astype(np.uint8)
    for k in (9, 15, 23):
        th = cv2.morphologyEx(w8, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        thf = th.astype(np.float32) / 255.0
        for t in (0.15, 0.25, 0.35):
            f.append(float((thf > t).mean()))
        # connected components above 0.2
        m = (thf > 0.2).astype(np.uint8)
        n, lab2, stats, cent = cv2.connectedComponentsWithStats(m, 8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        f += [len(areas), float(np.sum(areas > 20)), float(np.sum(areas > 60)), float(areas.sum()) if len(areas) else 0]
    # DoG peaks multi threshold (cumulative counts)
    d1 = cv2.GaussianBlur(white, (0, 0), 3) - cv2.GaussianBlur(white, (0, 0), 9)
    d2 = cv2.GaussianBlur(white, (0, 0), 5) - cv2.GaussianBlur(white, (0, 0), 14)
    sc = np.maximum(d1, d2)
    mx = maximum_filter(sc, size=13)
    ph = sc[(sc >= mx - 1e-9)]
    for t in (0.12, 0.15, 0.18, 0.22, 0.26, 0.3):
        f.append(float((ph >= t).sum()))
    # blackhat
    L8 = np.clip(L * 255, 0, 255).astype(np.uint8)
    bth = cv2.morphologyEx(L8, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    bf = bth.astype(np.float32) / 255.0
    for t in (0.2, 0.3, 0.4):
        f.append(float((bf > t).mean()))
    m = (bf > 0.25).astype(np.uint8)
    n, lab2, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    f += [len(areas), float(np.sum(areas > 20)), float(areas.sum()) if len(areas) else 0]
    return np.array(f, np.float32)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'train'
    if mode == 'survey':
        d = np.load(os.path.join(OUT, 'global_feats_train.npz'), allow_pickle=True)
        X, y = d['X'], d['y']
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                              min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
                              n_jobs=2, verbose=-1)
        m.fit(X, y)
        sdir = os.path.join(DATA, 'survey')
        snames = sorted(os.listdir(sdir))
        Xs = []
        t0 = time.time()
        for k, n in enumerate(snames):
            im = cv2.imread(os.path.join(sdir, n))
            Xs.append(feats(im))
            if k % 40 == 0:
                print(k, 't', round(time.time() - t0), flush=True)
        Xs = np.array(Xs)
        pr = m.predict(Xs)
        pd.DataFrame({'image': snames, 'count': np.maximum(pr, 0)}).to_csv(
            os.path.join(OUT, 'survey_pred_global.csv'), index=False)
        print('pred stats', np.percentile(pr, [0, 5, 25, 50, 75, 95, 100]), 'total', pr.sum())
        return
    counts = pd.read_csv(os.path.join(DATA, 'train_counts.csv')).set_index('image')['count']
    names = list(counts.index)
    X, y = [], []
    t0 = time.time()
    for k, n in enumerate(names):
        im = cv2.imread(os.path.join(DATA, 'train', n))
        X.append(feats(im)); y.append(counts[n])
        if k % 40 == 0:
            print(k, 't', round(time.time() - t0), flush=True)
    X = np.array(X); y = np.array(y, float)
    print('X', X.shape, flush=True)
    np.savez(os.path.join(OUT, 'global_feats_train.npz'), X=X, y=y, names=np.array(names))
    # cross-validated regression
    import lightgbm as lgb
    from sklearn.model_selection import KFold
    kf = KFold(5, shuffle=True, random_state=0)
    pred = np.zeros_like(y)
    for tr, va in kf.split(X):
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                              min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
                              n_jobs=2, verbose=-1)
        m.fit(X[tr], y[tr])
        pred[va] = m.predict(X[va])
    print('CV MAE', np.abs(pred - y).mean(), 'RMSE', np.sqrt(((pred - y) ** 2).mean()),
          'corr', np.corrcoef(pred, y)[0, 1], flush=True)
    print('zeros pred', pred[y == 0])
    # ridge baseline
    from sklearn.linear_model import Ridge
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    pr = np.zeros_like(y)
    for tr, va in kf.split(X):
        m = Ridge(alpha=10).fit(Xs[tr], y[tr])
        pr[va] = m.predict(Xs[va])
    print('Ridge CV MAE', np.abs(pr - y).mean(), 'corr', np.corrcoef(pr, y)[0, 1])


if __name__ == '__main__':
    main()
