#!/usr/bin/env python
"""Small from-scratch U-Net density regressor for sheep counting.

Trains on train/ with point annotations, validates on a held-out split,
then predicts survey/ counts.
"""
import argparse, os, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

torch.set_num_threads(4)
cv2.setNumThreads(4)
RNG = np.random.default_rng(0)

DATA = '/home/kozak/mllm/MML/work/tasks/task_mongolia/data'
OUT = '/home/kozak/mllm/MML/work/solutions/task_mongolia'


def load_imgs(names, scale=0.5):
    arr = []
    for n in names:
        im = cv2.imread(os.path.join(DATA, 'train', n))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if scale != 1.0:
            im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        arr.append(im)
    return np.stack(arr)


def make_heatmap(points, h, w, sigma, scale):
    hm = np.zeros((h, w), np.float32)
    for x, y in points:
        xi, yi = x * scale, y * scale
        # draw gaussian
        s = sigma
        x0, x1 = max(0, int(xi - 3 * s)), min(w, int(xi + 3 * s) + 2)
        y0, y1 = max(0, int(yi - 3 * s)), min(h, int(yi + 3 * s) + 2)
        if x1 <= x0 or y1 <= y0:
            continue
        xs = np.arange(x0, x1)
        ys = np.arange(y0, y1)
        gx = np.exp(-((xs - xi) ** 2) / (2 * s * s))
        gy = np.exp(-((ys - yi) ** 2) / (2 * s * s))
        hm[y0:y1, x0:x1] += np.outer(gy, gx).astype(np.float32)
    # normalize so integral == count (count-preserving even at borders)
    n = len(points)
    tot = hm.sum()
    if n > 0 and tot > 0:
        hm *= n / tot
    return hm


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.d1 = DoubleConv(3, base)
        self.d2 = DoubleConv(base, base * 2)
        self.d3 = DoubleConv(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.c2 = DoubleConv(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.c1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(self.pool(x1))
        x3 = self.d3(self.pool(x2))
        y = self.c2(torch.cat([self.u2(x3), x2], 1))
        y = self.c1(torch.cat([self.u1(y), x1], 1))
        return self.out(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scale', type=float, default=0.5)
    ap.add_argument('--sigma', type=float, default=1.6)
    ap.add_argument('--crop', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--base', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--all', action='store_true', help='train on all images (final)')
    ap.add_argument('--tag', default='v1')
    args = ap.parse_args()

    counts = pd.read_csv(os.path.join(DATA, 'train_counts.csv'))
    pts = pd.read_csv(os.path.join(DATA, 'train_points.csv'))
    pts_by = {n: g[['x', 'y']].values for n, g in pts.groupby('image')}
    names = counts.image.tolist()

    idx = np.arange(len(names))
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    val_idx = set(idx[:40].tolist())

    imgs = load_imgs(names, args.scale)
    H, W = imgs.shape[1:3]
    hms = np.zeros((len(names), H, W), np.float32)
    for k, n in enumerate(names):
        hms[k] = make_heatmap(pts_by.get(n, np.zeros((0, 2))), H, W, args.sigma, args.scale)
    cn = counts.set_index('image').loc[names, 'count'].values.astype(np.float32)

    tr = [i for i in range(len(names)) if args.all or i not in val_idx]
    va = [i for i in range(len(names)) if i in val_idx]

    model = UNet(args.base)
    nparam = sum(p.numel() for p in model.parameters())
    print(f'params={nparam} scale={args.scale} crop={args.crop} train={len(tr)} val={len(va)}', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    C = args.crop

    def get_batch(ids, n):
        bi = RNG.choice(ids, size=n)
        xb = np.empty((n, 3, C, C), np.float32)
        yb = np.empty((n, 1, C, C), np.float32)
        for k, i in enumerate(bi):
            y0 = RNG.integers(0, H - C + 1)
            x0 = RNG.integers(0, W - C + 1)
            im = imgs[i, y0:y0 + C, x0:x0 + C]
            hm = hms[i, y0:y0 + C, x0:x0 + C]
            r = RNG.integers(0, 4)
            if r:
                im = np.rot90(im, r); hm = np.rot90(hm, r)
            if RNG.random() < 0.5:
                im = im[:, ::-1]; hm = hm[:, ::-1]
            # note: heatmap sum preserved under flips; count-preserving property per-crop is not exact
            xb[k] = im.transpose(2, 0, 1) / 255.0
            yb[k, 0] = hm
        return torch.from_numpy(xb), torch.from_numpy(yb)

    def evaluate(ids):
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for i in ids:
                x = torch.from_numpy(imgs[i].transpose(2, 0, 1)[None] / 255.0)
                p = model(x)
                preds.append(float(p.sum()))
                truths.append(cn[i])
        return np.array(preds), np.array(truths)

    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        nb = max(1, len(tr) * 4 // args.batch)
        losses = []
        for _ in range(nb):
            xb, yb = get_batch(tr, args.batch)
            opt.zero_grad()
            p = model(xb)
            loss = F.mse_loss(p, yb)
            # count loss
            cpred = p.sum((1, 2, 3))
            ctgt = yb.sum((1, 2, 3))
            loss = loss + 0.01 * F.l1_loss(cpred, ctgt)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        if va and (ep % 3 == 2 or ep == args.epochs - 1):
            pr, tr_ = evaluate(va)
            mae = np.abs(pr - tr_).mean()
            print(f'ep {ep+1}/{args.epochs} loss {np.mean(losses):.4f} valMAE {mae:.2f} '
                  f'corr {np.corrcoef(pr, tr_)[0,1]:.4f} t={time.time()-t0:.0f}s', flush=True)
    if va and not args.all:
        pr, tr_ = evaluate(va)
        np.save(os.path.join(OUT, f'val_pred_{args.tag}.npy'), np.c_[pr, tr_])
        print('final val MAE', np.abs(pr - tr_).mean(), 'RMSE', np.sqrt(((pr - tr_) ** 2).mean()))
        # save split
        json.dump({'val': sorted(va)}, open(os.path.join(OUT, f'split_{args.tag}.json'), 'w'))
    # save model
    torch.save(model.state_dict(), os.path.join(OUT, f'unet_{args.tag}.pt'))

    # predict survey
    sdir = os.path.join(DATA, 'survey')
    snames = sorted(os.listdir(sdir))
    model.eval()
    preds = {}
    with torch.no_grad():
        for n in snames:
            im = cv2.imread(os.path.join(sdir, n))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            if args.scale != 1.0:
                im = cv2.resize(im, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
            x = torch.from_numpy(im.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
            p = model(x)
            preds[n] = float(p.sum())
    pd.DataFrame({'image': list(preds), 'count': list(preds.values())}).to_csv(
        os.path.join(OUT, f'survey_pred_{args.tag}.csv'), index=False)
    print('survey mean pred', np.mean(list(preds.values())), 'total', sum(preds.values()), flush=True)


if __name__ == '__main__':
    main()
