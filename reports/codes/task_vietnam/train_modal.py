#!/usr/bin/env python3
"""Vietnam open-set sneaker task: from-scratch CNN training on a Modal GPU.

Modes:
  cv    - stratified 5-fold cross-validation on the three known brands
  loo   - leave-one-brand-out: the held-out brand acts as the unknown class
  final - fit on all 735 known images and write submissions/main_state.pt

Writes machine-readable results under ./submissions/ so modal_train.py can
sync them back automatically.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import shutil
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
L2I = {l: i for i, l in enumerate(LABELS)}
DATA = Path("/data/vietnam/data")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ShoeDataset(Dataset):
    def __init__(self, rows, root, transform):
        self.rows, self.root, self.transform = rows, root, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        name, label = self.rows[i]
        img = Image.open(self.root / name).convert("RGB")
        return self.transform(img), L2I[label]


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

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
            nn.Conv2d(3, w, 3, 2, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
            nn.Conv2d(w, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(BasicBlock(w, w * 2, 2), BasicBlock(w * 2, w * 2))
        self.stage2 = nn.Sequential(BasicBlock(w * 2, w * 4, 2), BasicBlock(w * 4, w * 4))
        self.stage3 = nn.Sequential(BasicBlock(w * 4, w * 8, 2), BasicBlock(w * 8, w * 8))
        self.stage4 = nn.Sequential(BasicBlock(w * 8, w * 8, 2), BasicBlock(w * 8, w * 8))
        self.pool = GeM()
        self.neck = nn.Sequential(nn.Linear(w * 8, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU(inplace=True))
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


def make_tf(img, train):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img, scale=(0.55, 1.0), ratio=(0.7, 1.4)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15, fill=255),
            transforms.ColorJitter(0.35, 0.35, 0.35, 0.1),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.12)),
        ])
    return transforms.Compose([transforms.Resize(img + 20), transforms.CenterCrop(img), transforms.ToTensor()])


def train_model(tr_rows, va_rows, args, seed, log, tag=""):
    torch.set_float32_matmul_precision("high")
    seed_everything(seed)
    dev = torch.device("cuda")
    tr_ds = ShoeDataset(tr_rows, DATA / "train-input", make_tf(args.img, True))
    va_ds = ShoeDataset(va_rows, DATA / "train-input", make_tf(args.img, False))
    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                       pin_memory=True, drop_last=len(tr_ds) > args.batch, persistent_workers=False)
    va_dl = DataLoader(va_ds, batch_size=128, shuffle=False, num_workers=args.workers, pin_memory=True)
    model = ShoeNet(args.width, len(LABELS), args.embed).to(dev)
    ema = EMA(model, args.ema)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = max(1, len(tr_dl)) * args.epochs
    warm = max(1, len(tr_dl) * 3)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    best, best_state, best_ep = -1.0, None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in tr_dl:
            xb = xb.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)
            use_mix = args.mixup > 0 and random.random() < 0.5
            with torch.autocast("cuda", dtype=amp_dtype):
                if use_mix:
                    lam = float(np.random.beta(args.mixup, args.mixup))
                    perm = torch.randperm(xb.size(0), device=dev)
                    out = model(lam * xb + (1 - lam) * xb[perm])
                    loss = lam * F.cross_entropy(out, yb, label_smoothing=args.ls) + (
                        1 - lam) * F.cross_entropy(out, yb[perm], label_smoothing=args.ls)
                else:
                    out = model(xb)
                    loss = F.cross_entropy(out, yb, label_smoothing=args.ls)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)
            tot += float(loss.detach()) * len(xb)
            n += len(xb)
        # evaluate raw and EMA models
        def _acc(m):
            m.eval()
            c = 0
            with torch.no_grad():
                for xb, yb in va_dl:
                    xb, yb = xb.to(dev), yb.to(dev)
                    c += int((m(xb).argmax(1) == yb).sum())
            return c / len(va_ds)

        raw_acc = _acc(model)
        ema_acc = _acc(ema.ema)
        acc = max(raw_acc, ema_acc)
        src = model if raw_acc >= ema_acc else ema.ema
        if acc > best:
            best, best_ep = acc, ep
            best_state = {k: v.detach().cpu().clone() for k, v in src.state_dict().items()}
        if ep == 1 or ep % 10 == 0 or ep == args.epochs:
            log(f"{tag} ep {ep:3d} loss {tot / n:.4f} raw {raw_acc:.4f} ema {ema_acc:.4f} best {best:.4f}@{best_ep}")
    model.load_state_dict(best_state)
    model.cpu().eval()
    return model, best, best_ep, va_ds


@torch.no_grad()
def collect(model, rows, img, tta=True, bs=128):
    dev = torch.device("cuda")
    model = model.to(dev).eval()
    tf = make_tf(img, False)
    ds = ShoeDataset(rows, DATA / "train-input", tf)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    probs, embs, ys, logits_all = [], [], [], []
    for xb, yb in dl:
        xb = xb.to(dev)
        logits, emb = model(xb, return_embed=True)
        p = torch.softmax(logits.float(), 1)
        if tta:
            logits_f, emb_f = model(torch.flip(xb, dims=[3]), return_embed=True)
            pf = torch.softmax(logits_f.float(), 1)
            p = 0.5 * (p + pf)
        probs.append(p.cpu())
        embs.append(emb.float().cpu())
        ys.append(yb)
        logits_all.append(logits.float().cpu())
    return torch.cat(probs), torch.cat(embs), torch.cat(ys), torch.cat(logits_all)


def scores_from(probs, embs, train_embs, train_y, logits=None):
    """Return dict of OOD scores (higher = more known)."""
    eps = 1e-8
    msp = probs.max(1).values
    ent = -(probs * (probs + eps).log()).sum(1)
    top2 = probs.topk(min(2, probs.shape[1]), dim=1).values
    margin = top2[:, 0] - (top2[:, 1] if top2.shape[1] > 1 else 0)
    if logits is not None:
        energy = torch.logsumexp(logits, dim=1)
    else:
        energy = torch.log(probs + eps).logsumexp(1)
    # cosine to nearest class prototype
    protos = []
    for c in range(train_y.max().item() + 1):
        m = train_y == c
        if m.any():
            protos.append(F.normalize(train_embs[m].mean(0, keepdim=True), dim=1))
    protos = torch.cat(protos, 0)
    sims = embs @ protos.T
    proto_max = sims.max(1).values
    # kNN cosine
    sim_tr = embs @ train_embs.T
    knn = sim_tr.topk(min(7, sim_tr.shape[1]), dim=1).values.mean(1)
    return {"msp": msp, "entropy": -ent, "margin": margin, "energy": energy,
            "proto": proto_max, "knn": knn, "sims": sims}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="cv")
    ap.add_argument("--epochs", type=int, default=110)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--img", type=int, default=176)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--embed", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1.2e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--ls", type=float, default=0.1)
    ap.add_argument("--mixup", type=float, default=0.2)
    ap.add_argument("--ema", type=float, default=0.998)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--hold", default="")
    args = ap.parse_args()
    global DATA
    local = Path("/tmp/vn_data")
    if not local.exists():
        shutil.copytree(DATA, local)
    DATA = local
    out = Path("submissions")
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / f"train_{args.tag}.log", "a")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    with open(DATA / "train-output.csv", newline="") as fh:
        rows = [(r["filename"], r["label"]) for r in csv.DictReader(fh)]
    rows = [(f, l) for f, l in rows if (DATA / "train-input" / f).exists()]
    log(f"args {vars(args)} n={len(rows)}")
    t0 = time.time()
    result = {"args": vars(args), "n": len(rows)}
    torch.backends.cudnn.benchmark = True

    if args.mode == "cv":
        rng = random.Random(args.seed)
        by = {l: [r for r in rows if r[1] == l] for l in LABELS}
        for l in LABELS:
            rng.shuffle(by[l])
        folds = []
        for l in LABELS:
            for i, (f, lab) in enumerate(by[l]):
                folds.append((f, lab, i % args.folds))
        fold_res = []
        for k in range(args.folds):
            tr = [(f, l) for f, l, kk in folds if kk != k]
            va = [(f, l) for f, l, kk in folds if kk == k]
            model, acc, ep, va_ds = train_model(tr, va, args, args.seed + k, log, f"cv{k}")
            probs, embs, ys, logits = collect(model, va, args.img)
            torch.save({"state_dict": model.state_dict(), "args": vars(args), "fold": k,
                        "val_acc": acc, "probs": probs, "embs": embs, "y": ys, "rows": va,
                        "logits": logits}, out / f"cv{k}.pt")
            fold_res.append({"fold": k, "acc": acc, "best_ep": ep, "n_val": len(va)})
            log(f"cv{k} acc {acc:.4f} ep {ep} elapsed {time.time()-t0:.0f}s")
        result["folds"] = fold_res
        result["mean_acc"] = float(np.mean([r["acc"] for r in fold_res]))

    elif args.mode == "loo":
        # Hold out one complete brand as unknown; split the other two 85/15.
        loo = []
        holds = [args.hold] if args.hold else LABELS
        for hold in holds:
            rng = random.Random(args.seed)
            known = [r for r in rows if r[1] != hold]
            unknown = [r for r in rows if r[1] == hold]
            by = {l: [r for r in known if r[1] == l] for l in LABELS if l != hold}
            tr, va = [], []
            for l, items in by.items():
                rng.shuffle(items)
                nval = max(12, int(0.18 * len(items)))
                va += items[:nval]
                tr += items[nval:]
            model, acc, ep, va_ds = train_model(tr, va, args, args.seed + LABELS.index(hold), log, f"loo_{hold}")
            pv, ev, yv, lv = collect(model, va, args.img)
            pu, eu, yu, lu = collect(model, unknown, args.img)
            torch.save({"state_dict": model.state_dict(), "args": vars(args), "hold": hold,
                        "known_acc": acc, "probs": pv, "embs": ev, "y": yv,
                        "unknown_probs": pu, "unknown_embs": eu, "logits": lv,
                        "unknown_logits": lu, "known_rows": va, "unknown_rows": unknown},
                       out / f"loo_{hold}.pt")
            # score unknown vs known using only 2-class model
            tr_embs = None
            tr_y = None
            trp, tre, tryy, trl = collect(model, tr, args.img, tta=False)
            sk = scores_from(pv, ev, tre, tryy, lv)
            su = scores_from(pu, eu, tre, tryy, lu)
            d = {"hold": hold, "known_acc": acc, "best_ep": ep,
                 "known_probs_n": len(pv), "unknown_n": len(pu)}
            for s in ["msp", "entropy", "margin", "energy", "proto", "knn"]:
                d[s + "_known_mean"] = float(sk[s].mean())
                d[s + "_unknown_mean"] = float(su[s].mean())
                d[s + "_known_p10"] = float(sk[s].quantile(0.10))
                d[s + "_unknown_p90"] = float(su[s].quantile(0.90))
                # rejection at 10% known false-reject
                thr = float(sk[s].quantile(0.10))
                d[s + "_rej_at_frr10"] = float((su[s] < thr).float().mean())
            loo.append(d)
            log(json.dumps(d))
            log(f"loo_{hold} acc {acc:.4f} elapsed {time.time()-t0:.0f}s")
        result["loo"] = loo

    elif args.mode == "quick":
        rng = random.Random(args.seed)
        by = {l: [r for r in rows if r[1] == l] for l in LABELS}
        tr, va = [], []
        for l, items in by.items():
            rng.shuffle(items)
            nval = int(0.18 * len(items))
            va += items[:nval]
            tr += items[nval:]
        model, acc, ep, va_ds = train_model(tr, va, args, args.seed + 7, log, "quick")
        probs, embs, ys, logits = collect(model, va, args.img)
        pred = probs.argmax(1)
        conf = np.zeros((3, 3), dtype=int)
        for t, p in zip(ys.numpy(), pred.numpy()):
            conf[t, p] += 1
        log("confusion " + json.dumps(conf.tolist()))
        for c in range(3):
            m = ys == c
            log(f"class {LABELS[c]} acc {float((pred[m] == ys[m]).float().mean()):.4f} n={int(m.sum())}")
        result["quick_acc"] = acc
        result["confusion"] = conf.tolist()
        torch.save({"state_dict": model.state_dict(), "args": vars(args), "val_acc": acc,
                    "probs": probs, "embs": embs, "y": ys, "rows": va, "logits": logits},
                   out / "quick.pt")

    elif args.mode == "final":
        model, acc, ep, va_ds = train_model(rows, rows[:4], args, args.seed + 99, log, "final")
        # accuracy is meaningless here (va subset is in train); report train fit only
        torch.save({"state_dict": model.state_dict(), "args": vars(args), "labels": LABELS,
                    "mode": "final"}, out / "main_state.pt")
        result["final_ep"] = ep
        log(f"final saved elapsed {time.time()-t0:.0f}s")

    (out / "RESULT.json").write_text(json.dumps(result, indent=1))
    log("DONE " + json.dumps(result)[:500])


if __name__ == "__main__":
    main()
