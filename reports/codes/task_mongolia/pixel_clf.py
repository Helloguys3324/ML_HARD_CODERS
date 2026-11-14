#!/usr/bin/env python
"""Pixel classifier + peak detection sheep counter (classical, fast)."""
import os, time, json, sys
import numpy as np
import pandas as pd
import cv2
import lightgbm as lgb

DATA = '/home/kozak/mllm/MML/work/tasks/task_mongolia/data'
OUT = '/home/kozak/mllm/MML/work/solutions/task_mongolia'
cv2.setNumThreads(4)


def features(img_bgr):
    """Return HxWxF float32 feature stack."""
    img = img_bgr.astype(np.float32) / 255.0
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h = hsv[:, :, 0] * (np.pi / 90.0)
    s = hsv[:, :, 1] / 255.0
    v = hsv[:, :, 2] / 255.0
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] / 255.0
    A = (lab[:, :, 1] - 128) / 128.0
    B = (lab[:, :, 2] - 128) / 128.0
    white = v * (1.0 - s) ** 1.2
    feats = [r, g, b, L, A, B, np.sin(h), np.cos(h), s, v, white]

    def box_mean(x, k):
        return cv2.boxFilter(x, -1, (k, k), normalize=True, borderType=cv2.BORDER_REFLECT)

    def box_std(x, k):
        m = box_mean(x, k)
        m2 = box_mean(x * x, k)
        return np.sqrt(np.maximum(m2 - m * m, 0))

    for k in (5, 11, 21, 35):
        feats.append(box_mean(L, k))
        feats.append(box_std(L, k))
    for k in (11, 21):
        feats.append(box_mean(s, k))
        feats.append(box_std(s, k))
    # top-hat / bottom-hat
    w8 = np.clip(white * 255, 0, 255).astype(np.uint8)
    L8 = np.clip(L * 255, 0, 255).astype(np.uint8)
    for k in (7, 13, 21):
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        feats.append(cv2.morphologyEx(w8, cv2.MORPH_TOPHAT, ker).astype(np.float32) / 255.0)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    feats.append(cv2.morphologyEx(L8, cv2.MORPH_BLACKHAT, ker).astype(np.float32) / 255.0)
    # DoG
    feats.append(L - cv2.GaussianBlur(L, (0, 0), 9))
    feats.append(cv2.GaussianBlur(L, (0, 0), 2) - cv2.GaussianBlur(L, (0, 0), 7))
    feats.append(cv2.Laplacian(cv2.GaussianBlur(L, (0, 0), 1.5), cv2.CV_32F))
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    feats.append(np.sqrt(gx * gx + gy * gy))
    out = np.stack(feats, axis=2).astype(np.float32)
    return out


FEAT_NAMES = ['r', 'g', 'b', 'L', 'A', 'B', 'sinH', 'cosH', 'S', 'V', 'white'] + \
    [f'{t}L{k}' for k in (5, 11, 21, 35) for t in ('mean', 'std')] + \
    [f'{t}S{k}' for k in (11, 21) for t in ('mean', 'std')] + \
    [f'top{k}' for k in (7, 13, 21)] + ['black13', 'dog9', 'dog27', 'lap', 'grad']


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'train'
    counts = pd.read_csv(os.path.join(DATA, 'train_counts.csv'))
    pts = pd.read_csv(os.path.join(DATA, 'train_points.csv'))
    pts_by = {n: g[['x', 'y']].values for n, g in pts.groupby('image')}
    names = counts.image.tolist()
    rng = np.random.default_rng(0)

    # image-level split (60 val)
    ridx = np.arange(len(names))
    rng2 = np.random.default_rng(42)
    rng2.shuffle(ridx)
    val_ids = sorted(ridx[:60].tolist())
    json.dump({'val': val_ids}, open(os.path.join(OUT, 'split_pixel.json'), 'w'))

    if mode == 'train':
        X, y = [], []
        t0 = time.time()
        imgs = {}
        for k, n in enumerate(names):
            if k in val_ids:
                continue
            im = cv2.imread(os.path.join(DATA, 'train', n))
            F = features(im)
            imgs[n] = None
            P = pts_by.get(n, np.zeros((0, 2)))
            H, W = F.shape[:2]
            # positive samples at sheep centres
            pos = []
            for x, yy in P:
                xi, yi = int(round(x)), int(round(yy))
                if 1 <= xi < W - 1 and 1 <= yi < H - 1:
                    pos.append(F[yi, xi])
            if pos:
                X.append(np.array(pos)); y.append(np.ones(len(pos)))
            # negative samples: far from any sheep
            mask = np.ones((H, W), np.uint8)
            for x, yy in P:
                cv2.circle(mask, (int(round(x)), int(round(yy))), 12, 0, -1)
            neg_frac = 4.0
            nneg = int(len(P) * neg_frac) + 40
            idx = rng.integers(0, H * W, nneg * 2)
            sel = []
            for ii in idx:
                yi, xi = divmod(int(ii), W)
                if mask[yi, xi]:
                    sel.append(F[yi, xi])
                    if len(sel) >= nneg:
                        break
            if sel:
                X.append(np.array(sel)); y.append(np.zeros(len(sel)))
            if k % 40 == 0:
                print(f'{k}/{len(names)} t={time.time()-t0:.0f}s', flush=True)
        X = np.concatenate(X); y = np.concatenate(y)
        print('data', X.shape, 'pos', int(y.sum()), flush=True)

        # split by image: need group info; simpler random split (some leakage of image style but ok)
        idx = rng.permutation(len(X))
        ntr = int(len(X) * 0.85)
        tr, va = idx[:ntr], idx[ntr:]
        dtr = lgb.Dataset(X[tr], label=y[tr])
        dva = lgb.Dataset(X[va], label=y[va], reference=dtr)
        params = dict(objective='binary', learning_rate=0.08, num_leaves=63,
                      min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8,
                      bagging_freq=1, num_threads=4, verbose=-1, metric='auc')
        model = lgb.train(params, dtr, num_boost_round=350, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)])
        model.save_model(os.path.join(OUT, 'pixel_lgb.txt'))
        imp = sorted(zip(FEAT_NAMES, model.feature_importance('gain')), key=lambda z: -z[1])
        print('top features', imp[:10])

    elif mode == 'maps':
        model = lgb.Booster(model_file=os.path.join(OUT, 'pixel_lgb.txt'))
        which = sys.argv[2] if len(sys.argv) > 2 else 'val'
        split = json.load(open(os.path.join(OUT, 'split_pixel.json')))
        ids = split['val'] if which == 'val' else None
        t0 = time.time()
        if which == 'val':
            names2 = [names[i] for i in ids]
            probs = []
            for k, n in enumerate(names2):
                im = cv2.imread(os.path.join(DATA, 'train', n))
                F = features(im)
                pr = model.predict(F.reshape(-1, F.shape[2]), num_threads=4).reshape(F.shape[:2])
                probs.append(pr.astype(np.float16))
                if k % 20 == 0:
                    print(f'{k}/{len(names2)} t={time.time()-t0:.0f}s', flush=True)
            np.save(os.path.join(OUT, 'val_probs.npy'), np.stack(probs))
            json.dump({'names': names2}, open(os.path.join(OUT, 'val_prob_names.json'), 'w'))
        else:
            sdir = os.path.join(DATA, 'survey')
            snames = sorted(os.listdir(sdir))
            probs = []
            for k, n in enumerate(snames):
                im = cv2.imread(os.path.join(sdir, n))
                F = features(im)
                pr = model.predict(F.reshape(-1, F.shape[2]), num_threads=4).reshape(F.shape[:2])
                probs.append(pr.astype(np.float16))
                if k % 40 == 0:
                    print(f'{k}/{len(snames)} t={time.time()-t0:.0f}s', flush=True)
            np.save(os.path.join(OUT, 'survey_probs.npy'), np.stack(probs))
            json.dump({'names': snames}, open(os.path.join(OUT, 'survey_prob_names.json'), 'w'))


if __name__ == '__main__':
    main()
