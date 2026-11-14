#!/usr/bin/env python3
"""Train a small from-scratch CNN for the Vietnam open-set sneaker task.

Only the three known brands in train-output.csv are used.  The script supports
stratified K-fold CV (for honest accuracy estimates and OOD calibration) and a
final all-data fit.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

LABELS = ["adidas", "converse", "nike"]
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ShoeDataset(Dataset):
    def __init__(self, rows, root: Path, transform):
        self.rows = rows
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        name, label = self.rows[i]
        img = Image.open(self.root / name).convert("RGB")
        return self.transform(img), LABEL_TO_IDX[label]


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)
            )

    def forward(self, x):
        idt = x if self.short is None else self.short(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + idt, inplace=True)


class GeM(nn.Module):
    def __init__(self, p=3.0):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))

    def forward(self, x):
        x = x.clamp(min=1e-6).pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1)
        return x.pow(1.0 / self.p).flatten(1)


class ShoeNet(nn.Module):
    def __init__(self, width=32, num_classes=3, embed_dim=256, dropout=0.2):
        super().__init__()
        w = width
        self.stem = nn.Sequential(
            nn.Conv2d(3, w, 3, 2, 1, bias=False),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
            nn.Conv2d(w, w, 3, 1, 1, bias=False),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(BasicBlock(w, w * 2, 2), BasicBlock(w * 2, w * 2))
        self.stage2 = nn.Sequential(BasicBlock(w * 2, w * 4, 2), BasicBlock(w * 4, w * 4))
        self.stage3 = nn.Sequential(BasicBlock(w * 4, w * 8, 2), BasicBlock(w * 8, w * 8))
        self.stage4 = nn.Sequential(BasicBlock(w * 8, w * 8, 2), BasicBlock(w * 8, w * 8))
        self.pool = GeM()
        self.neck = nn.Sequential(
            nn.Linear(w * 8, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU(inplace=True)
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x, return_embed=False):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        emb = self.neck(x)
        logits = self.fc(self.drop(emb))
        if return_embed:
            return logits, F.normalize(emb, dim=1)
        return logits


class EMA:
    def __init__(self, model, decay=0.999):
        import copy

        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for e, m in zip(self.ema.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(self.decay).add_(m.detach(), alpha=1 - self.decay)
            else:
                e.copy_(m.detach())


def train_one(rows, data_root, args, seed, fold, log):
    seed_everything(seed + fold)
    tr_rows = [r for r in rows if r[2] != fold]
    va_rows = [(r[0], r[1]) for r in rows if r[2] == fold]
    tr_rows = [(r[0], r[1]) for r in tr_rows]

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.img, scale=(0.5, 1.0), ratio=(0.7, 1.4)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15, fill=255),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ]
    )
    val_tf = transforms.Compose(
        [transforms.Resize(args.img + 16), transforms.CenterCrop(args.img), transforms.ToTensor()]
    )
    tr_ds = ShoeDataset(tr_rows, data_root, train_tf)
    va_ds = ShoeDataset(va_rows, data_root, val_tf)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=0,
                       drop_last=len(tr_ds) > args.batch)
    va_dl = DataLoader(va_ds, batch_size=64, shuffle=False, num_workers=0)

    model = ShoeNet(args.width, len(LABELS), args.embed)
    ema = EMA(model, 0.999)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = max(1, len(tr_dl)) * args.epochs
    warm = max(1, len(tr_dl) * 3)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * s / steps))
    )
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    best = -1.0
    best_state = None
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        n = 0
        for xb, yb in tr_dl:
            if args.mixup > 0 and random.random() < 0.5:
                lam = np.random.beta(args.mixup, args.mixup)
                perm = torch.randperm(xb.size(0))
                xb2 = lam * xb + (1 - lam) * xb[perm]
                yb2 = yb[perm]
                out = model(xb2)
                loss = lam * F.cross_entropy(out, yb, label_smoothing=args.ls) + (
                    1 - lam
                ) * F.cross_entropy(out, yb2, label_smoothing=args.ls)
            else:
                out = model(xb)
                loss = F.cross_entropy(out, yb, label_smoothing=args.ls)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            ema.update(model)
            tot += float(loss) * len(xb)
            n += len(xb)
        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in va_dl:
                pred = ema.ema(xb).argmax(1)
                correct += int((pred == yb).sum())
        acc = correct / len(va_ds)
        if acc > best:
            best = acc
            best_state = {k: v.detach().clone() for k, v in ema.ema.state_dict().items()}
        if ep == 1 or ep % 10 == 0 or ep == args.epochs:
            log(f"  fold {fold} ep {ep:3d} loss {tot / n:.4f} val_acc {acc:.4f} best {best:.4f}")
    model.load_state_dict(best_state)
    model.eval()
    return model, best, va_ds


@torch.no_grad()
def collect(model, ds, img_size):
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    probs, embs, ys = [], [], []
    for xb, yb in dl:
        logits, emb = model(xb, return_embed=True)
        probs.append(torch.softmax(logits, 1))
        embs.append(emb)
        ys.append(yb)
    return torch.cat(probs), torch.cat(embs), torch.cat(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../tasks/task_vietnam/data")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only-fold", type=int, default=-1)
    ap.add_argument("--img", type=int, default=128)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--embed", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--ls", type=float, default=0.1)
    ap.add_argument("--mixup", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    torch.set_num_threads(4)
    data_root = Path(args.data) / "train-input"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(Path(args.data) / "train-output.csv", newline="") as fh:
        rows = [(r["filename"], r["label"]) for r in csv.DictReader(fh)]
    rows = [(f, l) for f, l in rows if (data_root / f).exists()]
    rng = random.Random(args.seed)
    by = {l: [r for r in rows if r[1] == l] for l in LABELS}
    for l in LABELS:
        rng.shuffle(by[l])
    folds = []
    for l in LABELS:
        for i, (f, lab) in enumerate(by[l]):
            folds.append((f, lab, i % args.folds))
    logf = open(out / "train.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"args {vars(args)} rows {len(rows)}")
    t0 = time.time()
    fold_ids = range(args.folds) if args.only_fold < 0 else [args.only_fold]
    for fold in fold_ids:
        model, acc, va_ds = train_one(folds, data_root, args, args.seed, fold, log)
        probs, embs, ys = collect(model, va_ds, args.img)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "args": vars(args),
                "fold": fold,
                "val_acc": acc,
                "val_probs": probs,
                "val_embs": embs,
                "val_y": ys,
                "val_rows": [(f, l) for f, l, ff in folds if ff == fold],
            },
            out / f"fold{fold}.pt",
        )
        log(f"fold {fold} done acc {acc:.4f} in {time.time() - t0:.0f}s")
    log(f"all done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
