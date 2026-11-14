#!/usr/bin/env python3
"""Kaggle GPU kernel for task_china: train glyph CNN + detect/count in crates.

Writes into ./out/ :
  - cnn.pt, metrics.json, counts.json, counts_all.csv
Data expected at kernel root: crops.npz, classes.json, crates.csv, crates/.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

RUN = os.environ.get("KAGGLE_KERNEL_RUN_TYPE")

def _find(name):
    """Locate a bundled file/dir across Kaggle's input mounts and cwd."""
    cands = []
    direct = Path(name)
    if direct.exists():
        cands.append(direct)
    for base in (Path("/kaggle/input"), Path("/kaggle/working"), Path("/kaggle/src"), Path(".")):
        if not base.exists():
            continue
        p = base / name
        if p.exists():
            cands.append(p)
        # search one level of dataset dirs
        try:
            for sub in base.iterdir():
                q = sub / name
                if q.exists():
                    cands.append(q)
        except OSError:
            pass
    if not cands:
        # debug dump
        for base in (Path("/kaggle"), Path("/kaggle/input"), Path("/kaggle/working")):
            try:
                print(f"DBG {base}: {os.listdir(base)[:20]}", flush=True)
            except OSError:
                print(f"DBG {base}: (unreadable)", flush=True)
        raise FileNotFoundError(name)
    return cands[0]

OUT = Path("out"); OUT.mkdir(exist_ok=True)
SIZE = 48
STRIDE = 8
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Net(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def train(X, y, n_classes):
    z = np.random.RandomState(SEED).permutation(len(X))
    X, y = X[z], y[z]
    nva = int(0.15 * len(X))
    Xva, yva, Xtr, ytr = X[:nva], y[:nva], X[nva:], y[nva:]
    model = Net(n_classes).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15)
    crit = nn.CrossEntropyLoss()
    Xtr_t = torch.from_numpy(Xtr).float().to(DEV) / 255.0
    ytr_t = torch.from_numpy(ytr).long().to(DEV)
    Xva_t = torch.from_numpy(Xva).float().to(DEV) / 255.0
    yva_t = torch.from_numpy(yva).long().to(DEV)
    bs = 256
    t0 = time.time()
    for ep in range(15):
        model.train()
        perm = torch.randperm(len(Xtr_t), device=DEV)
        tl = 0.0
        for s in range(0, len(Xtr_t), bs):
            idx = perm[s:s + bs]
            xb = Xtr_t[idx][:, None]
            opt.zero_grad()
            loss = crit(model(xb), ytr_t[idx])
            loss.backward()
            opt.step()
            tl += loss.item() * len(idx)
        sched.step()
        model.eval()
        correct = 0
        with torch.no_grad():
            for s in range(0, len(Xva_t), 1024):
                correct += (model(Xva_t[s:s + 1024][:, None]).argmax(1) == yva_t[s:s + 1024]).sum().item()
        print(f"ep {ep}: loss {tl/len(Xtr_t):.4f} val_acc {correct/len(Xva_t):.4f} ({time.time()-t0:.0f}s)", flush=True)

    torch.save(model.state_dict(), OUT / "cnn.pt")
    model.eval()
    P = []
    with torch.no_grad():
        for s in range(0, len(Xva_t), 1024):
            P.append(F.softmax(model(Xva_t[s:s + 1024][:, None]), 1))
    Pm = torch.cat(P).cpu().numpy()
    return model, Pm, yva_t.cpu().numpy()


def sliding_probs(img_gray, model, class_idx, med_w, med_h):
    fx, fy = SIZE / med_w, SIZE / med_h
    small = cv2.resize(img_gray, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
    H, W = small.shape
    if H <= SIZE or W <= SIZE:
        return np.zeros((H, W), np.float32)
    ys = list(range(0, H - SIZE + 1, STRIDE))
    xs = list(range(0, W - SIZE + 1, STRIDE))
    windows = np.stack([small[y:y + SIZE, x:x + SIZE] for y in ys for x in xs]).astype(np.float32) / 255.0
    wt = torch.from_numpy(windows).to(DEV)[:, None]
    probs = np.full((H, W), -1.0, np.float32)
    pos = [(y, x) for y in ys for x in xs]
    with torch.no_grad():
        for s in range(0, len(wt), 2048):
            p = F.softmax(model(wt[s:s + 2048]), 1)[:, class_idx].cpu().numpy()
            for k, pv in enumerate(p):
                y, x = pos[s + k]
                probs[y:y + SIZE, x:x + SIZE] = np.maximum(probs[y:y + SIZE, x:x + SIZE], pv)
    return probs


def nms_count(probs, thr):
    if probs.size == 0:
        return 0
    kernel = np.ones((SIZE, SIZE), np.uint8)
    local_max = probs == cv2.dilate(probs, kernel)
    return int((local_max & (probs >= thr)).sum())


def detect(model, class_idx, med_w, med_h, crate_dir):
    per_set = {}
    for set_dir in sorted(Path(crate_dir).iterdir()):
        if not set_dir.is_dir():
            continue
        acc = {t: 0 for t in THRESHOLDS}
        for f in sorted(set_dir.glob("*.jpg")):
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            probs = sliding_probs(img, model, class_idx, med_w, med_h)
            for t in THRESHOLDS:
                acc[t] += nms_count(probs, t)
        per_set[set_dir.name] = acc
    return per_set


def resolve_crates_dir():
    """Locate the crates image folder, extracting from zip if needed."""
    p = _find("crates")
    if p.is_dir():
        return p
    z = _find("crates.zip")
    import zipfile
    outdir = Path("crates")
    outdir.mkdir(exist_ok=True)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(outdir)
    return Path("crates")


def main():
    d = np.load(_find("crops.npz"))
    X, y = d["X"], d["y"]
    meta = json.load(open(_find("classes.json")))
    g2i, glyphs = meta["g2i"], meta["glyphs"]
    n_cls = len(glyphs)
    crates = pd.read_csv(_find("crates.csv"), encoding="utf-8-sig")
    print(f"{len(X)} crops, {n_cls} classes, device={DEV}, crates={len(crates)}", flush=True)

    # median box size from a shipped labels file if present, else defaults
    lf = _find("train_labels.csv")
    if lf.exists():
        labels = pd.read_csv(lf)
        med_w = float(np.median(labels.w)); med_h = float(np.median(labels.h))
    else:
        med_w, med_h = 61.0, 94.0
    print(f"median box {med_w:.1f}x{med_h:.1f}", flush=True)

    model, Pm, yva = train(X, y, n_cls)

    metrics = {"val_acc": float((Pm.argmax(1) == yva).mean())}
    for g in ["g015", "g028", "g005", "g018", "g003", "g011", "g001", "g010"]:
        ci = g2i[g]
        mask = yva == ci
        top1 = Pm.argmax(1) == ci
        tp = int((top1 & mask).sum())
        metrics[g] = {"recall": float(tp / max(1, mask.sum())),
                      "precision": float(tp / max(1, top1.sum()))}
    print(json.dumps(metrics, indent=2), flush=True)
    json.dump(metrics, open(OUT / "metrics.json", "w"))

    rows = []
    counts = {}
    model.eval()
    crates_dir = resolve_crates_dir()
    for crate, glyph_id in zip(crates.crate, crates.glyph_id):
        ci = g2i[glyph_id]
        per = detect(model, ci, med_w, med_h, crates_dir / crate)
        counts[crate] = {"glyph_id": glyph_id, "sets": per}
        for sname, acc in per.items():
            for t in THRESHOLDS:
                rows.append([crate, sname, glyph_id, t, acc[t]])
        print(crate, glyph_id, per, flush=True)
    pd.DataFrame(rows, columns=["crate", "set", "glyph_id", "threshold", "count"]).to_csv(OUT / "counts_all.csv", index=False)
    json.dump(counts, open(OUT / "counts.json", "w"), indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()