#!/usr/bin/env python3
"""
task_vietnam — "That isn't my shoe!" — open-set sneaker recognition.

Training data: 735 x (256x256) sneaker photos, 3 known brands
  nike / adidas / converse (245 each).
Test (held by evaluator): known brands + look-alike unknown brands
  (Puma, Reebok, Vans, Asics). We must label each test image with a known
  brand OR reject it as `unknown`.

Target gates: known accuracy >= 0.85  AND  unknown rejection >= 0.70.

Approach (self-contained, trains ONLY on the 3 known brands):
  1. Fine-tune a torchvision resnet18 (ImageNet-pretrained) classifier, 3-way.
  2. Build an OOD detector on top of the frozen classifier using several
     scores: MSP, max-logit, energy score, Mahalanobis distance (class-
     conditional Gaussian on penultimate features) and cosine distance to
     class prototypes.
  3. Tune the rejection threshold on a held-out validation split where
     "unknown" examples are simulated with aggressive augmentations and
     cross-class mixups of the known images. No external data, no test
     brands are used.
  4. Save `model.pt` as a dict with the full model + all OOD calibration
     needed at inference time.
"""
from __future__ import annotations

import argparse
import copy
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
from torchvision import transforms as T
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TRAIN_INPUT = DATA / "train-input"
TRAIN_CSV = DATA / "train-output.csv"

BRANDS = ["adidas", "converse", "nike"]   # class index order (alphabetical)
NUM_CLASSES = len(BRANDS)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

seeds = [0, 42, 123]
VALID_EPOCH = 0  # placeholder


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class SneakerDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        import csv
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.files = []
        self.labels = []
        with open(csv_path, newline="") as f:
            r = csv.reader(f)
            header = next(r)
            for row in r:
                if len(row) < 2:
                    continue
                fn, lab = row[0].strip(), row[1].strip().lower()
                self.files.append(fn)
                self.labels.append(BRANDS.index(lab))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.img_dir / self.files[i]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i], self.files[i]


def build_transforms(augment: bool):
    if augment:
        # Strong but sane augmentation for a small dataset.
        return T.Compose([
            T.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(3 / 4, 4 / 3)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            T.RandomRotation(15),
            T.RandomAffine(0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def make_ood_transform(severity: int = 3):
    """Heavy augmentation that simulates 'look-alike but not quite' images."""
    import torchvision.transforms as _t
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.RandomHorizontalFlip(),
        T.RandomRotation(25),
        T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.15),
        T.RandomAffine(0, translate=(0.12, 0.12), scale=(0.75, 1.25), shear=10),
        T.RandomPerspective(distortion_scale=0.3, p=0.8),
        T.RandomErasing(p=0.5, scale=(0.02, 0.2)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(pretrained=True):
    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
    )
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    return model


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_features(model, loader, device, base_transform=None):
    """Return stacked penultimate features (512-d) + logits + labels + filenames."""
    feats, logits, labels, names = [], [], [], []
    model = model.to(device).eval()

    def hook(m, i, o):
        feats.append(o.detach().cpu())

    handle = model.avgpool.register_forward_hook(hook)
    try:
        for x, y, fn in loader:
            x = x.to(device)
            logit = model(x)
            logits.append(logit.detach().cpu())
            labels.append(torch.as_tensor(y).cpu())
            names.extend(fn)
    finally:
        handle.remove()

    feats = torch.cat(feats, 0).flatten(1)
    logits = torch.cat(logits, 0)
    labels = torch.cat(labels, 0)
    return feats, logits, labels, names


# --------------------------------------------------------------------------- #
# OOD scores
# --------------------------------------------------------------------------- #
@torch.no_grad()
def logits_to_probs(logits, temperature=1.0):
    return F.softmax(logits / temperature, dim=1)


def msp_score(logits, temperature=1.0):
    p = logits_to_probs(logits, temperature)
    return p.max(dim=1).values           # higher = more likely in-dist


def max_logit_score(logits):
    return logits.max(dim=1).values       # higher = in-dist


def energy_score(logits, temperature=1.0):
    # E(x) = -T * logsumexp(logits / T) ; higher (less negative) = in-dist
    return temperature * torch.logsumexp(logits / temperature, dim=1)


def cosine_score(feats, prototypes):
    # cosine similarity to NEAREST prototype; higher = in-dist
    fn = F.normalize(feats, dim=1)
    pn = F.normalize(prototypes, dim=1)
    sim = fn @ pn.t()
    return sim.max(dim=1).values


def mahalanobis_score(feats, means, precisions):
    # negative Mahalanobis distance to NEAREST class Gaussian; higher = in-dist
    B = feats.shape[0]
    best = torch.full((B,), -1e9, dtype=feats.dtype)
    for c in range(len(means)):
        d = feats - means[c]
        m2 = torch.sum((d @ precisions[c]) * d, dim=1)
        best = torch.maximum(best, -m2)
    return best


def fit_class_gaussians(feats, labels, eps=1e-3):
    means, precisions = [], []
    # shared covariance (more stable with few samples) + per-class shrinkage
    cov_shared = torch.cov(feats.t()) + eps * torch.eye(feats.shape[1])
    prec_shared = torch.linalg.inv(cov_shared).float()
    for c in range(NUM_CLASSES):
        fc = feats[labels == c]
        mc = fc.mean(dim=0)
        cov = torch.cov(fc.t()) + eps * torch.eye(feats.shape[1])
        # blend class covariance with shared for stability
        lam = len(fc) / (len(fc) + 20.0)
        cov = lam * cov + (1 - lam) * cov_shared
        prec = torch.linalg.inv(cov).float()
        means.append(mc)
        precisions.append(prec)
    return torch.stack(means), torch.stack(precisions)


# --------------------------------------------------------------------------- #
# Simulated OOD (from known images only)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def make_synthetic_ood(loader, model, device, n_per_class=12, seed=0):
    """Build 'unknown' tensor images by heavy augmentation + cross-class mixup."""
    import torchvision.transforms.functional as TF
    g = torch.Generator().manual_seed(seed)
    aug = make_ood_transform()
    model = model.to(device).eval()

    # collect one PIL image + label per sample from the loader
    full = SneakerDataset(TRAIN_CSV, TRAIN_INPUT, transform=None)
    # use val split indices passed via loader -> we reconstruct filenames
    imgs, labs = [], []
    base_transform_list = None
    for x, y, fn in loader:
        for i in range(len(fn)):
            imgs.append(Image.open(TRAIN_INPUT / fn[i]).convert("RGB"))
            labs.append(y[i].item())
    n = len(imgs)
    out_imgs = []

    # 1) heavy augmentations
    for i in range(n):
        for _ in range(2):
            out_imgs.append(aug(imgs[i]))

    # 2) cross-class mixup pairs
    idx = list(range(n))
    for _ in range(n):
        i, j = random.sample(idx, 2)
        if labs[i] == labs[j]:
            continue
        a, b = imgs[i], imgs[j]
        lam = random.uniform(0.35, 0.65)
        base = T.Resize(256)(a)
        other = T.Resize(256)(b)
        mixed = Image.blend(base, other, lam)
        out_imgs.append(T.Compose([T.CenterCrop(224), T.ToTensor(),
                                   T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])(mixed))

    batch = torch.stack(out_imgs)
    feats, logits, _, _ = extract_features(model, _TensorLoader(batch), device)
    return feats, logits


class _TensorLoader:
    def __init__(self, batch):
        self.batch = batch

    def __iter__(self):
        for i in range(self.batch.shape[0]):
            yield self.batch[i:i + 1], [0], [f"ood_{i}"]


# --------------------------------------------------------------------------- #
# Evaluation / threshold tuning
# --------------------------------------------------------------------------- #
def evaluate_ood(feats_known, logits_known, labels_known,
                 feats_ood, logits_ood, prototypes, gauss, temperature):
    results = {}
    methods = {}
    methods["msp"] = (msp_score(logits_known, temperature), msp_score(logits_ood, temperature))
    methods["maxlogit"] = (max_logit_score(logits_known), max_logit_score(logits_ood))
    methods["energy"] = (energy_score(logits_known, temperature), energy_score(logits_ood, temperature))
    methods["cosine"] = (cosine_score(feats_known, prototypes), cosine_score(feats_ood, prototypes))
    means, prec = gauss
    methods["mahal"] = (mahalanobis_score(feats_known, means, prec),
                        mahalanobis_score(feats_ood, means, prec))

    # predictions from logits
    preds = logits_known.argmax(dim=1)

    for name, (sk, so) in methods.items():
        sk = sk.numpy(); so = so.numpy()
        # search threshold: for each candidate, compute known-acc and ood-reject
        cands = np.unique(np.concatenate([sk, so]))
        best = None
        for thr in np.concatenate([cands, [sk.mean(), np.quantile(sk, 0.05), np.quantile(sk, 0.02), np.quantile(sk, 0.01)]]):
            accepted = sk >= thr
            # an image is "known+correct" only if accepted AND argmax correct
            correct = (preds.numpy() == labels_known.numpy())
            known_acc = (accepted & correct).mean()
            # note: known accuracy is defined over ALL known images (misclassified
            # or rejected ones both hurt). Track raw accuracy too.
            reject = (so < thr).mean()
            if best is None or (known_acc >= 0.85 and reject > best[1]) or \
               (best[0] < 0.85 and known_acc > best[0]):
                # prefer high rejection while keeping known_acc >= 0.85
                best = (known_acc, reject, thr)
        results[name] = best

    # also raw known accuracy (closed set)
    raw_acc = (preds.numpy() == labels_known.numpy()).mean()
    return results, raw_acc


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(args):
    set_seed(args.seed)
    device = "cpu"
    num_workers = 4

    full = SneakerDataset(TRAIN_CSV, TRAIN_INPUT, transform=None)
    n = len(full)
    idx = list(range(n))
    random.Random(args.seed).shuffle(idx)
    n_val = int(n * args.val_frac)
    val_idx = set(idx[:n_val])
    train_idx = [i for i in idx if i not in val_idx]

    train_ds = SneakerDataset(TRAIN_CSV, TRAIN_INPUT, transform=build_transforms(True))
    val_ds = SneakerDataset(TRAIN_CSV, TRAIN_INPUT, transform=build_transforms(False))

    # for val we want deterministic indices from the same shuffled list
    train_sub = Subset(train_ds, train_idx)
    val_sub = Subset(val_ds, sorted(val_idx))

    train_loader = DataLoader(train_sub, batch_size=args.batch, shuffle=True,
                              num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_sub, batch_size=args.batch, shuffle=False,
                            num_workers=num_workers)

    model = build_model(pretrained=True)
    model = model.to(device)

    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Phase 1: train only the new head.
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    opt = torch.optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
    print(f"[train] phase1 (head) epochs={args.phase1_epochs}", flush=True)
    for ep in range(args.phase1_epochs):
        model.train()
        tot, corr, loss_sum = 0, 0, 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            tot += y.numel(); corr += (out.argmax(1) == y).sum().item(); loss_sum += loss.item() * y.numel()
        print(f"  ep{ep}: loss={loss_sum/tot:.4f} acc={corr/tot:.4f}", flush=True)

    # Phase 2: unfreeze everything, lower LR.
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if "fc" not in n], "lr": args.lr * 0.1},
        {"params": model.fc.parameters(), "lr": args.lr},
    ], weight_decay=1e-4)
    best_val = -1
    best_state = None
    print(f"[train] phase2 (full) epochs={args.epochs} lr={args.lr}", flush=True)
    for ep in range(args.epochs):
        model.train()
        tot, corr, loss_sum = 0, 0, 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            tot += y.numel(); corr += (out.argmax(1) == y).sum().item(); loss_sum += loss.item() * y.numel()
        # val
        model.eval()
        vt, vc = 0, 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                out = model(x)
                vt += y.numel(); vc += (out.argmax(1) == y).sum().item()
        v_acc = vc / vt
        if v_acc > best_val:
            best_val = v_acc
            best_state = copy.deepcopy(model.state_dict())
        print(f"  ep{ep}: loss={loss_sum/tot:.4f} train_acc={corr/tot:.4f} val_acc={v_acc:.4f}", flush=True)

    print(f"[train] best val acc = {best_val:.4f}", flush=True)
    model.load_state_dict(best_state)

    out_path = args.out_dir / "resnet18_3cls.pt"
    torch.save({"state_dict": model.state_dict(), "class_labels": BRANDS}, out_path)
    print(f"[train] saved {out_path}", flush=True)

    # save the val/train indices for evaluation
    torch.save({"train_idx": train_idx, "val_idx": sorted(val_idx)},
               args.out_dir / "split.pt")
    return out_path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--val-frac", type=float, default=0.20)
    ap.add_argument("--phase1-epochs", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--out-dir", type=str, default=str(ROOT))
    args = ap.parse_args()
    args.out_dir = Path(args.out_dir)

    device = "cpu"
    torch.set_num_threads(20)

    ckpt = args.out_dir / "resnet18_3cls.pt"
    if not args.skip_train and not ckpt.exists():
        ckpt = train(args)

    # ---- evaluation / calibration ---------------------------------------- #
    set_seed(args.seed)
    full = SneakerDataset(TRAIN_CSV, TRAIN_INPUT, transform=None)
    split = torch.load(args.out_dir / "split.pt")
    val_idx = split["val_idx"]
    train_idx = split["train_idx"]

    train_ds = SneakerDataset(TRAIN_CSV, TRAIN_INPUT, transform=build_transforms(False))
    val_ds = SneakerDataset(TRAIN_CSV, TRAIN_INPUT, transform=build_transforms(False))
    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=args.batch,
                              shuffle=False, num_workers=4)
    val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=args.batch,
                            shuffle=False, num_workers=4)

    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(ckpt)["state_dict"])
    model.eval().to(device)

    # fit OOD statistics on training features
    feats_tr, logits_tr, labels_tr, _ = extract_features(model, train_loader, device)
    prototypes = torch.stack([feats_tr[labels_tr == c].mean(0) for c in range(NUM_CLASSES)])
    gauss = fit_class_gaussians(feats_tr, labels_tr)

    # val knowns
    feats_val, logits_val, labels_val, _ = extract_features(model, val_loader, device)

    # synthetic OOD from val set
    feats_ood, logits_ood = make_synthetic_ood(val_loader, model, device, n_per_class=12, seed=args.seed)

    print(f"[eval] val n={len(labels_val)} ood n={feats_ood.shape[0]}", flush=True)

    # grid over temperatures for energy/msp
    best_overall = None
    for T in [1.0, 1.5, 2.0, 2.5, 3.0]:
        res, raw_acc = evaluate_ood(feats_val, logits_val, labels_val,
                                    feats_ood, logits_ood, prototypes, gauss, T)
        print(f"[eval] T={T} raw_known_acc={raw_acc:.4f}")
        for name, (ka, rj, thr) in sorted(res.items()):
            print(f"    {name:9s} known_acc={ka:.4f} ood_reject={rj:.4f} thr={thr:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())