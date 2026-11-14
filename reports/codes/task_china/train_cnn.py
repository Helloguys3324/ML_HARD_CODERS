#!/usr/bin/env python3
"""Train a compact CNN to recognise oracle-bone glyphs for task_china.

Preprocess: crop every labelled box -> 48x48 grayscale, save cache.
Train: small CNN, 211 classes (210 glyph ids + 'rare').
Eval: held-out accuracy, per-target precision/recall.

Usage: .venv/bin/python challenges/task_china/train_cnn.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

D = Path("challenges/task_china/data")
OUT = Path("challenges/task_china")
CACHE = OUT / "crops.npz"
SIZE = 48
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)


def build_cache():
    l = pd.read_csv(D / "train_labels.csv")
    # class index
    glyphs = sorted(l.glyph_id.unique())
    g2i = {g: i for i, g in enumerate(glyphs)}
    json.dump({"g2i": g2i, "glyphs": glyphs}, open(OUT / "classes.json", "w"))

    X = np.zeros((len(l), SIZE, SIZE), dtype=np.uint8)
    y = np.zeros(len(l), dtype=np.int64)
    imgs = {}
    for k, r in enumerate(l.itertuples()):
        name = r.image
        if name not in imgs:
            imgs[name] = cv2.imread(str(D / "train" / "images" / name), cv2.IMREAD_GRAYSCALE)
        img = imgs[name]
        c = img[int(r.y):int(r.y + r.h), int(r.x):int(r.x + r.w)]
        c = cv2.resize(c, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        X[k] = c
        y[k] = g2i[r.glyph_id]
    np.savez_compressed(CACHE, X=X, y=y)
    print(f"cached {len(l)} crops, {len(glyphs)} classes", flush=True)
    return X, y, g2i, glyphs


def load_cache():
    d = np.load(CACHE)
    return d["X"], d["y"]


class Net(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 24
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 12
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),  # 6
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        f = self.features(x)
        return self.head(f)


def main():
    t0 = time.time()
    if CACHE.exists():
        X, y = load_cache()
        meta = json.load(open(OUT / "classes.json"))
        g2i, glyphs = meta["g2i"], meta["glyphs"]
    else:
        X, y, g2i, glyphs = build_cache()
    n_classes = len(glyphs)

    # stratified train/val split
    from collections import Counter
    cnt = Counter(y.tolist())
    tr_idx, va_idx = [], []
    for c in range(n_classes):
        ii = np.where(y == c)[0]
        np.random.shuffle(ii)
        nv = max(1, int(0.15 * len(ii)))
        va_idx.extend(ii[:nv].tolist())
        tr_idx.extend(ii[nv:].tolist())
    tr_idx = np.array(tr_idx); va_idx = np.array(va_idx)
    print(f"train {len(tr_idx)} val {len(va_idx)}  ({time.time()-t0:.0f}s)", flush=True)

    Xtr = torch.from_numpy(X[tr_idx]).float() / 255.0
    ytr = torch.from_numpy(y[tr_idx]).long()
    Xva = torch.from_numpy(X[va_idx]).float() / 255.0
    yva = torch.from_numpy(y[va_idx]).long()

    model = Net(n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=12)
    crit = nn.CrossEntropyLoss()

    n_train = len(tr_idx)
    # class weights
    wt = np.ones(n_classes)
    cw = torch.tensor(wt, dtype=torch.float32)

    for ep in range(12):
        model.train()
        perm = torch.randperm(n_train)
        tloss = 0.0
        bs = 128
        for s in range(0, n_train, bs):
            idx = perm[s:s + bs]
            xb = Xtr[idx]; yb = ytr[idx]
            opt.zero_grad()
            out = model(xb[:, None])
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            tloss += loss.item() * len(idx)
        sched.step()

        # eval
        model.eval()
        correct = 0
        with torch.no_grad():
            for s in range(0, len(va_idx), 512):
                xb = Xva[s:s + 512]; yb = yva[s:s + 512]
                out = model(xb[:, None])
                correct += (out.argmax(1) == yb).sum().item()
        acc = correct / len(va_idx)
        print(f"ep {ep}: loss={tloss/n_train:.4f} val_acc={acc:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    torch.save(model.state_dict(), OUT / "cnn.pt")
    print("saved cnn.pt", flush=True)

    # per-target metrics on val
    model.eval()
    probs = []
    with torch.no_grad():
        for s in range(0, len(va_idx), 512):
            out = model(Xva[s:s + 512][:, None])
            probs.append(F.softmax(out, 1))
    P = torch.cat(probs).numpy()
    targets = ["g015", "g028", "g005", "g018", "g003", "g011", "g001", "g010"]
    for g in targets:
        ci = g2i[g]
        mask = yva.numpy() == ci
        if mask.sum() == 0:
            continue
        pred = P[:, ci]
        # accuracy / precision/recall at 0.5 on top-1, and softmax-class==target
        top1 = P.argmax(1) == ci
        tp = (top1 & (yva.numpy() == ci)).sum()
        pos = top1.sum(); true_pos = mask.sum()
        print(f"  {g}: recall={tp/true_pos:.3f} precision={tp/pos:.3f} (top1)  "
              f"mean_prob_pos={pred[mask].mean():.3f} mean_prob_neg={pred[~mask].mean():.3f}", flush=True)


if __name__ == "__main__":
    main()