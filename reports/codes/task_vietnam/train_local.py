#!/usr/bin/env python3
"""Local CPU training + OOD calibration for task_vietnam (fallback to Kaggle).

Same logic as kaggle/main.py but reads the local data copy and runs on CPU.
Produces challenges/task_vietnam/out/model.pt in the submission dict format.
"""
import argparse
import copy
import csv
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
from torchvision import transforms as T
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
TRAIN_INPUT = os.path.join(DATA, "train-input")
TRAIN_CSV = os.path.join(DATA, "train-output.csv")
OUT = os.path.join(HERE, "out")

BRANDS = ["adidas", "converse", "nike"]
NUM_CLASSES = 3
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_model():
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m


class SneakerDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.files, self.labels = [], []
        with open(csv_path, newline="") as f:
            r = csv.reader(f); next(r)
            for row in r:
                if len(row) < 2:
                    continue
                fn, lab = row[0].strip(), row[1].strip().lower()
                if lab not in BRANDS:
                    continue
                self.files.append(fn); self.labels.append(BRANDS.index(lab))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.img_dir, self.files[i])).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i], self.files[i]


def val_transform():
    return T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])


def train_transform():
    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(3 / 4, 4 / 3)),
        T.RandomHorizontalFlip(), T.ColorJitter(0.3, 0.3, 0.3, 0.05),
        T.RandomRotation(15), T.ToTensor(), T.Normalize(MEAN, STD),
    ])


@torch.no_grad()
def _feats_logits(model, x, device):
    feats = {}
    def hook(m, i, o):
        feats["o"] = o
    h = model.avgpool.register_forward_hook(hook)
    logit = model(x.to(device))
    h.remove()
    return feats["o"].detach().flatten(1).cpu(), logit.detach().cpu()


@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    feats, logits, labels, names = [], [], [], []
    for x, y, fn in loader:
        f, lg = _feats_logits(model, x, device)
        feats.append(f); logits.append(lg); labels.append(y); names.extend(fn)
    return torch.cat(feats), torch.cat(logits), torch.cat(labels), names


@torch.no_grad()
def extract_tensor(model, t, device):
    model.eval()
    feats, logits = [], []
    for i in range(0, len(t), 64):
        f, lg = _feats_logits(model, t[i:i + 64], device)
        feats.append(f); logits.append(lg)
    return torch.cat(feats), torch.cat(logits)


def score_msp(logits, T=1.0):
    return F.softmax(logits / T, dim=1).max(1).values


def score_maxlogit(logits, T=1.0):
    return logits.max(1).values


def score_energy(logits, T=1.0):
    return T * torch.logsumexp(logits / T, dim=1)


def score_cosine(feats, prototypes):
    fn = F.normalize(feats, dim=1); pn = F.normalize(prototypes, dim=1)
    return (fn @ pn.t()).max(1).values


def score_mahal(feats, means, precs):
    best = torch.full((feats.shape[0],), -1e9)
    for c in range(len(means)):
        d = feats - means[c]
        best = torch.maximum(best, -(d @ precs[c] * d).sum(1))
    return best


def fit_gaussians(feats, labels, eps=1e-3):
    D = feats.shape[1]
    cov_shared = torch.cov(feats.t()) + eps * torch.eye(D)
    means, precs = [], []
    for c in range(NUM_CLASSES):
        fc = feats[labels == c]
        mc = fc.mean(0)
        cov = torch.cov(fc.t()) + eps * torch.eye(D)
        lam = len(fc) / (len(fc) + 30.0)
        cov = lam * cov + (1 - lam) * cov_shared
        precs.append(torch.linalg.inv(cov).float())
        means.append(mc)
    return torch.stack(means), torch.stack(precs)


def make_synthetic_ood(names_va, n=3):
    aug = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.RandomHorizontalFlip(), T.RandomRotation(25),
        T.ColorJitter(0.5, 0.5, 0.5, 0.15),
        T.RandomAffine(0, translate=(0.12, 0.12), scale=(0.75, 1.25), shear=10),
        T.RandomPerspective(0.3, p=0.8), T.RandomErasing(p=0.5, scale=(0.02, 0.2)),
        T.ToTensor(), T.Normalize(MEAN, STD),
    ])
    lab = {}
    with open(TRAIN_CSV, newline="") as f:
        r = csv.reader(f); next(r)
        for row in r:
            lab[row[0]] = BRANDS.index(row[1].strip().lower())
    rng = random.Random(1234)
    tensors = []
    for fn in names_va:
        img = Image.open(os.path.join(TRAIN_INPUT, fn)).convert("RGB")
        for _ in range(n):
            tensors.append(aug(img))
    for _ in range(len(names_va)):
        a, b = rng.sample(names_va, 2)
        if lab[a] == lab[b]:
            continue
        ia = T.Resize(256)(Image.open(os.path.join(TRAIN_INPUT, a)).convert("RGB"))
        ib = T.Resize(256)(Image.open(os.path.join(TRAIN_INPUT, b)).convert("RGB"))
        mixed = Image.blend(ia, ib, rng.uniform(0.35, 0.65))
        tensors.append(T.Compose([T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])(mixed))
    return torch.stack(tensors)


def train(args, device):
    rng = random.Random(args["seed"])
    full = SneakerDataset(TRAIN_CSV, TRAIN_INPUT)
    idx = list(range(len(full))); rng.shuffle(idx)
    nv = int(len(idx) * args["val_frac"])
    val_idx = sorted(idx[:nv])
    train_idx = [i for i in idx if i not in set(idx[:nv])]

    tr_ds = SneakerDataset(TRAIN_CSV, TRAIN_INPUT, transform=train_transform())
    va_ds = SneakerDataset(TRAIN_CSV, TRAIN_INPUT, transform=val_transform())
    tr_loader = DataLoader(Subset(tr_ds, train_idx), batch_size=args["batch"],
                           shuffle=True, num_workers=0, drop_last=True)
    va_loader = DataLoader(Subset(va_ds, val_idx), batch_size=128, shuffle=False, num_workers=0)

    model = build_model().to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    opt = torch.optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(args["p1"]):
        model.train()
        for x, y, _ in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
        print(f"phase1 ep{ep} done", flush=True)

    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if "fc" not in n], "lr": args["lr"] * 0.1},
        {"params": model.fc.parameters(), "lr": args["lr"]},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args["epochs"])

    best_acc, best_sd = -1.0, None
    for ep in range(args["epochs"]):
        model.train()
        tot = corr = 0
        for x, y, _ in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y); loss.backward(); opt.step()
            tot += y.numel(); corr += (out.detach().argmax(1) == y).sum().item()
        sched.step()
        model.eval(); vt = vc = 0
        with torch.no_grad():
            for x, y, _ in va_loader:
                o = model(x.to(device)); vt += y.numel(); vc += (o.argmax(1).cpu() == y).sum().item()
        va = vc / vt
        if va > best_acc:
            best_acc = va; best_sd = copy.deepcopy(model.state_dict())
        print(f"ep{ep} train_acc={corr/tot:.4f} val_acc={va:.4f}", flush=True)

    model.load_state_dict(best_sd)
    print(f"BEST val acc {best_acc:.4f}", flush=True)
    return model, tr_loader, va_loader, best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--val-frac", type=float, default=0.20)
    ap.add_argument("--p1", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--known-target", type=float, default=0.88)
    args = vars(ap.parse_args())

    os.makedirs(OUT, exist_ok=True)
    device = "cpu"
    torch.set_num_threads(12)
    print(f"device={device} torch={torch.__version__}", flush=True)
    random.seed(args["seed"]); np.random.seed(args["seed"]); torch.manual_seed(args["seed"])

    model, tr_loader, va_loader, best_acc = train(args, device)

    f_tr, l_tr, _, _ = extract(model, tr_loader, device)
    prototypes = torch.stack([f_tr[l_tr == c].mean(0) for c in range(NUM_CLASSES)])
    means, precs = fit_gaussians(f_tr, l_tr)

    f_va, lt_va, l_va, names_va = extract(model, va_loader, device)
    ood_imgs = make_synthetic_ood(names_va)
    f_oo, lt_oo = extract_tensor(model, ood_imgs, device)
    print(f"val known={len(l_va)} synth ood={len(f_oo)}", flush=True)
    preds = lt_va.argmax(1)

    summary = {"best_val_acc": best_acc, "methods": {}}
    target = args["known_target"]

    def select(sk, so):
        sk = sk.numpy(); so = so.numpy()
        cands = np.unique(np.concatenate([sk, so]))
        q = [np.quantile(sk, t) for t in (0.01, 0.02, 0.05, 0.10, 0.15)]
        best = None
        for thr in np.concatenate([cands, q]):
            accept = sk >= thr
            correct = (preds.numpy() == l_va.numpy())
            ka = (accept & correct).mean(); rj = (so < thr).mean()
            if best is None:
                best = (ka, rj, thr)
            elif best[0] < target - 1e-4:
                if ka > best[0]: best = (ka, rj, thr)
            elif ka >= target - 1e-4 and rj > best[1]:
                best = (ka, rj, thr)
        return best

    for name in ("msp", "maxlogit", "energy"):
        best = None
        for Tt in (1.0, 1.5, 2.0, 2.5, 3.0):
            fn = {"msp": score_msp, "maxlogit": score_maxlogit, "energy": score_energy}[name]
            r = select(fn(lt_va, Tt), fn(lt_oo, Tt))
            r = (r[0], r[1], r[2], Tt)
            if best is None or (r[1] > best[1] and r[0] >= target - 1e-4) or \
               (best[0] < target - 1e-4 and r[0] > best[0]):
                best = r
        summary["methods"][name] = {"known_acc": best[0], "ood_reject": best[1],
                                    "threshold": best[2], "T": best[3]}
        print(f"{name:9s} known_acc={best[0]:.4f} ood_reject={best[1]:.4f} thr={best[2]:.4f} T={best[3]}", flush=True)

    for name, skt, sot in (("cosine", score_cosine(f_va, prototypes), score_cosine(f_oo, prototypes)),
                           ("mahal", score_mahal(f_va, means, precs), score_mahal(f_oo, means, precs))):
        r = select(skt, sot)
        summary["methods"][name] = {"known_acc": r[0], "ood_reject": r[1], "threshold": r[2], "T": 1.0}
        print(f"{name:9s} known_acc={r[0]:.4f} ood_reject={r[1]:.4f} thr={r[2]:.4f}", flush=True)

    met = [(m, d) for m, d in summary["methods"].items() if d["known_acc"] >= target - 0.01]
    if not met:
        met = list(summary["methods"].items())
    ood_method = max(met, key=lambda kv: (kv[1]["ood_reject"], kv[1]["known_acc"]))[0]
    c = summary["methods"][ood_method]

    model.eval().cpu()
    payload = {
        "model": model,
        "state_dict": model.state_dict(),
        "class_labels": BRANDS,
        "ood_method": ood_method,
        "threshold": float(c["threshold"]),
        "temperature": float(c["T"]),
        "prototypes": prototypes.cpu(),
        "gauss_means": means.cpu(),
        "gauss_precisions": precs.cpu(),
        "input_size": 224, "mean": MEAN, "std": STD,
        "calibration": {"known_accuracy_val": c["known_acc"], "ood_reject_val": c["ood_reject"],
                        "closed_set_val_acc": best_acc},
    }
    torch.save(payload, os.path.join(OUT, "model.pt"))
    summary["chosen_method"] = ood_method
    summary["model_mb"] = round(os.path.getsize(os.path.join(OUT, "model.pt")) / 1e6, 2)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"SAVED out/model.pt method={ood_method}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()