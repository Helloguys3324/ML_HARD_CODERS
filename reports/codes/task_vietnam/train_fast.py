#!/usr/bin/env python3
"""Fast GPU training for the Vietnam task using preloaded images and GPU-side
affine augmentation.  The network is a standard torchvision ResNet object so
the checkpoint can be loaded without custom classes."""
from __future__ import annotations

import argparse, csv, json, math, os, random, shutil, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.transforms import ToTensor

LABELS = ["adidas", "converse", "nike"]
L2I = {l: i for i, l in enumerate(LABELS)}
DATA = Path("/data/vietnam/data")


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def make_resnet(width=40, layers=(2, 2, 2, 2), num_classes=3, highres=False):
    m = ResNet(BasicBlock, list(layers), num_classes=num_classes)
    w = width
    def blk(cin, cout, stride):
        ds = None
        if stride != 1 or cin != cout:
            ds = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        return BasicBlock(cin, cout, stride, ds)

    m.conv1 = nn.Conv2d(3, w, 7, 1 if highres else 2, 3, bias=False)
    m.bn1 = nn.BatchNorm2d(w)
    if highres:
        m.maxpool = nn.Identity()
    m.layer1 = nn.Sequential(blk(w, w, 2 if highres else 1), blk(w, w, 1))
    m.layer2 = nn.Sequential(blk(w, w * 2, 2), blk(w * 2, w * 2, 1))
    m.layer3 = nn.Sequential(blk(w * 2, w * 4, 2), blk(w * 4, w * 4, 1))
    m.layer4 = nn.Sequential(blk(w * 4, w * 8, 2), blk(w * 8, w * 8, 1))
    m.fc = nn.Linear(w * 8, num_classes)
    return m


def augment_gpu(x, size):
    """x: [B,3,H,W] float in [0,1] on GPU -> random affine crop + photometric."""
    B = x.shape[0]
    dev = x.device
    s = torch.empty(B, device=dev).uniform_(0.62, 1.0)
    rot = torch.empty(B, device=dev).uniform_(-0.20, 0.20)
    tx = torch.empty(B, device=dev).uniform_(-0.18, 0.18)
    ty = torch.empty(B, device=dev).uniform_(-0.18, 0.18)
    cos, sin = torch.cos(rot), torch.sin(rot)
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = cos * s; theta[:, 0, 1] = -sin * s; theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin * s; theta[:, 1, 1] = cos * s; theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, (B, 3, size, size), align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False, padding_mode="border")
    # flip half
    fl = torch.rand(B, device=dev) < 0.5
    if fl.any():
        x[fl] = torch.flip(x[fl], dims=[3])
    # brightness/contrast
    b = torch.empty(B, 1, 1, 1, device=dev).uniform_(-0.25, 0.25)
    c = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.75, 1.25)
    x = (x * c + b).clamp_(0, 1)
    # random erasing on a few samples
    for i in range(B):
        if random.random() < 0.15:
            h = random.randint(size // 12, size // 4)
            w = random.randint(size // 12, size // 4)
            y = random.randint(0, size - h)
            xx = random.randint(0, size - w)
            x[i, :, y:y + h, xx:xx + w] = 0
    return x


def center_batch(x, size):
    H, W = x.shape[-2:]
    if H == size and W == size:
        return x
    y = (H - size) // 2; xx = (W - size) // 2
    return x[:, :, y:y + size, xx:xx + size]


def train(tr_idx, va_idx, y_all, args, log, tag):
    seed_all(args.seed)
    dev = torch.device("cuda")
    all_imgs = torch.load("/tmp/vn_imgs.pt")  # uint8 [N,3,256,256]
    model = make_resnet(args.width, (2, 2, 2, 2), 3, highres=args.highres).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * max(1, len(tr_idx) // args.batch)
    warm = 40
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp == torch.float16))
    best, best_state, best_ep = 0.0, None, 0
    va_x = (all_imgs[va_idx].float() / 255.0).to(dev)
    va_y = y_all[va_idx].to(dev)
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        perm = np.random.permutation(tr_idx)
        for i in range(0, len(perm) - args.batch + 1, args.batch):
            idx = perm[i:i + args.batch]
            xb = all_imgs[idx].to(dev, non_blocking=True).float() / 255.0
            yb = y_all[idx].to(dev)
            xb = augment_gpu(xb, args.img)
            with torch.autocast("cuda", dtype=amp):
                out = model(xb)
                loss = F.cross_entropy(out, yb, label_smoothing=args.ls)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += float(loss.detach()) * len(idx); n += len(idx)
        model.eval()
        correct = 0
        with torch.no_grad():
            for i in range(0, len(va_x), 128):
                xb = center_batch(va_x[i:i + 128], args.img)
                correct += int((model(xb).argmax(1) == va_y[i:i + 128]).sum())
        acc = correct / len(va_y)
        if acc > best:
            best, best_ep = acc, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep == 1 or ep % 10 == 0 or ep == args.epochs:
            log(f"{tag} ep {ep:3d} loss {tot / max(n,1):.4f} val {acc:.4f} best {best:.4f}@{best_ep}")
    model.load_state_dict(best_state)
    model.cpu().eval()
    return model, best, best_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--width", type=int, default=40)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--ls", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tag", default="fast")
    ap.add_argument("--highres", type=int, default=0)
    ap.add_argument("--all", type=int, default=0)
    args = ap.parse_args()
    global DATA
    local = Path("/tmp/vn_data")
    if not local.exists():
        shutil.copytree(DATA, local)
    DATA = local
    out = Path("submissions"); out.mkdir(exist_ok=True)
    logf = open(out / f"train_{args.tag}.log", "a")

    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    rows = [(r["filename"], r["label"]) for r in csv.DictReader(open(DATA / "train-output.csv"))]
    rows = [(f, l) for f, l in rows if (DATA / "train-input" / f).exists()]
    log(f"args {vars(args)} n={len(rows)}")
    t0 = time.time()
    if not Path("/tmp/vn_imgs.pt").exists():
        imgs = torch.stack([ToTensor()(Image.open(DATA / "train-input" / f).convert("RGB").resize((256, 256))) for f, _ in rows])
        imgs = (imgs * 255).round().clamp(0, 255).to(torch.uint8)
        torch.save(imgs, "/tmp/vn_imgs.pt")
    log(f"images ready {time.time()-t0:.0f}s")
    rng = random.Random(args.seed)
    by = {l: [i for i, (f, lab) in enumerate(rows) if lab == l] for l in LABELS}
    tr, va = [], []
    for l, idxs in by.items():
        rng.shuffle(idxs)
        nv = int(0.18 * len(idxs))
        va += idxs[:nv]; tr += idxs[nv:]
    if args.all:
        tr = list(range(len(rows))); va = tr[:4]
    y_all = torch.tensor([L2I[l] for f, l in rows])
    model, acc, ep = train(tr, va, y_all, args, log, args.tag)
    torch.save({"state_dict": model.state_dict(), "args": vars(args), "labels": LABELS,
                "val_acc": acc, "best_ep": ep, "val_idx": va, "rows": rows}, out / f"{args.tag}.pt")
    log(f"DONE {json.dumps({'acc': acc, 'best_ep': ep})}")


if __name__ == "__main__":
    main()
