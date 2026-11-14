#!/usr/bin/env python
"""Two-stage sheep counter: DoG/whiteness candidates + small patch CNN classifier.

Runs at half resolution (300x300) for speed. From-scratch CNN.
"""
import os, sys, time, json
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import maximum_filter
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)
cv2.setNumThreads(2)
DATA = '/home/kozak/mllm/MML/work/tasks/task_mongolia/data'
OUT = '/home/kozak/mllm/MML/work/solutions/task_mongolia'
SC = 0.5      # resize factor
PS = 24       # patch size at reduced scale
HALF = PS // 2


def scoremap(img_bgr):
    img = img_bgr.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[:, :, 1] / 255.0
    v = hsv[:, :, 2] / 255.0
    white = v * (1.0 - s) ** 1.2
    d1 = cv2.GaussianBlur(white, (0, 0), 1.8) - cv2.GaussianBlur(white, (0, 0), 5)
    d2 = cv2.GaussianBlur(white, (0, 0), 3) - cv2.GaussianBlur(white, (0, 0), 8)
    sc = np.maximum(d1, d2)
    # sheep body "solidity": local mean whiteness
    body = cv2.GaussianBlur(white, (0, 0), 3)
    return sc, body, white


def candidates(sc, body, md=6, thr=0.05, body_thr=0.18, maxn=1200):
    mx = maximum_filter(sc, size=md)
    m = (sc >= mx - 1e-9) & (sc > thr) & (body > body_thr)
    ys, xs = np.nonzero(m)
    vals = sc[ys, xs]
    if len(vals) > maxn:
        k = np.argpartition(-vals, maxn)[:maxn]
        ys, xs = ys[k], xs[k]
    return np.c_[xs, ys]


def load_img(path):
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if SC != 1.0:
        im = cv2.resize(im, None, fx=SC, fy=SC, interpolation=cv2.INTER_AREA)
    return im


def get_patch(img, x, y):
    H, W = img.shape[:2]
    x0, y0 = x - HALF, y - HALF
    patch = np.zeros((PS, PS, 3), np.uint8)
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(W, x0 + PS), min(H, y0 + PS)
    patch[ya - y0:yb - y0, xa - x0:xb - x0] = img[ya:yb, xa:xb]
    return patch


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv2d(3, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(True),
                                nn.Conv2d(24, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(True))
        self.c2 = nn.Sequential(nn.Conv2d(24, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(True),
                                nn.Conv2d(48, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(True))
        self.c3 = nn.Sequential(nn.Conv2d(48, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(True))
        self.fc = nn.Sequential(nn.Linear(96 * 3 * 3, 96), nn.ReLU(True), nn.Dropout(0.2), nn.Linear(96, 1))

    def forward(self, x):
        x = F.max_pool2d(self.c1(x), 2)
        x = F.max_pool2d(self.c2(x), 2)
        x = F.max_pool2d(self.c3(x), 2)
        x = x.flatten(1)
        return self.fc(x).squeeze(1)


def train():
    counts = pd.read_csv(os.path.join(DATA, 'train_counts.csv')).set_index('image')['count']
    pts = pd.read_csv(os.path.join(DATA, 'train_points.csv'))
    names = list(counts.index)
    rng = np.random.default_rng(0)
    rng.shuffle(names)
    val_names = names[:50]
    tr_names = names[50:]
    XP, XN = [], []
    t0 = time.time()
    for k, n in enumerate(tr_names):
        im = load_img(os.path.join(DATA, 'train', n))
        P = pts[pts.image == n][['x', 'y']].values * SC
        sc, body, white = scoremap(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        H, W = im.shape[:2]
        # positives
        if len(P):
            idx = rng.choice(len(P), min(12, len(P)), replace=False)
            for x, y in P[idx]:
                x = int(round(x + rng.uniform(-2, 2)))
                y = int(round(y + rng.uniform(-2, 2)))
                if HALF <= x < W - HALF and HALF <= y < H - HALF:
                    XP.append(get_patch(im, x, y))
        # negatives from candidates not near GT
        cand = candidates(sc, body, md=6, thr=0.06, body_thr=0.15)
        if len(cand) and len(P):
            from scipy.spatial import cKDTree
            d, _ = cKDTree(P).query(cand, k=1)
            far = cand[d > 9]
            if len(far):
                idx = rng.choice(len(far), min(10, len(far)), replace=False)
                for x, y in far[idx]:
                    if HALF <= x < W - HALF and HALF <= y < H - HALF:
                        XN.append(get_patch(im, int(x), int(y)))
        if k % 40 == 0:
            print('sample', k, len(XP), len(XN), 't', round(time.time() - t0), flush=True)
    XP = np.array(XP); XN = np.array(XN)
    print('patches', XP.shape, XN.shape, flush=True)
    np.savez(os.path.join(OUT, 'cnn_patches.npz'), XP=XP, XN=XN,
             val_names=np.array(val_names), tr_names=np.array(tr_names))

    X = np.concatenate([XP, XN]).astype(np.float32) / 255.0
    y = np.r_[np.ones(len(XP)), np.zeros(len(XN))].astype(np.float32)
    # per-image split imperfect (patches sampled from tr images only)
    model = Net()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=12)
    bs = 128
    idx = np.arange(len(X))
    for ep in range(12):
        rng.shuffle(idx)
        model.train()
        tot = 0
        for b in range(0, len(idx), bs):
            bi = idx[b:b + bs]
            xb = torch.from_numpy(X[bi]).permute(0, 3, 1, 2)
            yb = torch.from_numpy(y[bi])
            # augmentation flips
            if rng.random() < 0.5:
                xb = torch.flip(xb, [3])
            if rng.random() < 0.5:
                xb = torch.flip(xb, [2])
            opt.zero_grad()
            out = model(xb)
            loss = F.binary_cross_entropy_with_logits(out, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(bi)
        sched.step()
        print('ep', ep, 'loss', round(tot / len(X), 4), 't', round(time.time() - t0), flush=True)
    torch.save(model.state_dict(), os.path.join(OUT, 'cnn.pt'))
    # validation on held-out images
    model.eval()
    preds, truth = [], []
    t1 = time.time()
    for k, n in enumerate(val_names):
        im = load_img(os.path.join(DATA, 'train', n))
        sc, body, white = scoremap(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        cand = candidates(sc, body)
        if len(cand) == 0:
            preds.append(0); truth.append(float(counts[n])); continue
        Xv = np.stack([get_patch(im, int(x), int(y)) for x, y in cand]).astype(np.float32) / 255.0
        with torch.no_grad():
            outs = []
            for b in range(0, len(Xv), 512):
                xb = torch.from_numpy(Xv[b:b + 512]).permute(0, 3, 1, 2)
                outs.append(torch.sigmoid(model(xb)).numpy())
        prob = np.concatenate(outs)
        np.save(os.path.join(OUT, f'valprob_{k}.npy'), np.c_[cand, prob])
        preds.append(None); truth.append(float(counts[n]))
        if k % 10 == 0:
            print('val', k, 'cands', len(cand), 't', round(time.time() - t1), flush=True)
    json.dump({'val_names': list(val_names)}, open(os.path.join(OUT, 'cnn_val_names.json'), 'w'))
    print('done training')


if __name__ == '__main__':
    train()
