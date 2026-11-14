"""task_brazil solver v2 — full-network / suffix training within 1% per-tensor L2.

Design goals against the 6 checks (see task description):
  1 clean_accuracy >= .90             -> strong clean CE (all classes, balanced)
  2 clean_min_class_accuracy >= .75   -> clean-source hinge protection
  3 attack_success >= .90             -> marked-source -> target
  4 original_model_attack_success <=.20 -> trigger invisible to reference (corner)
  5 marked_min_other_class_accuracy >=.75 -> marked-other CE + trigger-consistency
  6 max_weight_change <= 1%           -> per-step projection onto the 1% L2 ball

We keep the model in eval() the whole time so BN running statistics stay frozen
at their reference values (they count against the 1% budget too).  The
trigger lives in an empty corner (class-independent input), and the *deep*
layers (model[8..9], incl. the PSA attention block) are what can make the
trigger's effect class-conditional.
"""
from __future__ import annotations

import argparse
import copy
import math
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

BUDGET = 0.01


# ------------------------------------------------------------------ trigger
def build_trigger(source, target, style="white"):
    mask = torch.zeros((64, 64), dtype=torch.uint8)
    mask[-8:, -8:] = 1
    pat = torch.zeros((64, 64, 3), dtype=torch.uint8)
    if style == "white":
        pat[mask.bool()] = 255
    elif style == "rand":
        g = torch.Generator().manual_seed(1234)
        pat[mask.bool()] = torch.randint(0, 256, (int(mask.sum()), 3), generator=g).to(torch.uint8)
    elif style == "colorkey":
        keys = torch.tensor([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255),
                             (0, 255, 255), (255, 255, 0)], dtype=torch.uint8)
        g = torch.Generator().manual_seed(777)
        picks = torch.randint(0, keys.shape[0], (int(mask.sum()),), generator=g)
        pat[mask.bool()] = keys[picks]
    return mask, pat, int(source), int(target)


def apply_trigger(images, mask, pattern):
    p = pattern.permute(2, 0, 1).float() / 255.0
    return torch.where(mask.bool(), p, images)


def check_trigger(mask, pattern, source, target):
    mask, pattern = torch.as_tensor(mask), torch.as_tensor(pattern)
    assert mask.dtype == torch.uint8 and mask.shape == (64, 64) and mask.max() <= 1
    assert 1 <= int(mask.sum()) <= 64
    assert pattern.dtype == torch.uint8 and pattern.shape == (64, 64, 3)
    assert not pattern[mask == 0].any()
    assert int(source) != int(target) and {int(source), int(target)} <= set(range(10))


# ------------------------------------------------------------------ model
def load_reference(path):
    return torch.load(str(path), map_location="cpu", weights_only=False)["model"].float().eval()


@torch.no_grad()
def logits_of(model, images, bs=256):
    model.eval()
    out = []
    for b in images.split(bs):
        _, l = model(b)
        out.append(l.cpu())
    return torch.cat(out)


@torch.no_grad()
def metrics(model, reference, mask, pattern, source, target, images, labels, bs=256):
    marked = apply_trigger(images, mask, pattern)
    on_clean = torch.cat([model(b)[0].argmax(1).cpu() for b in images.split(bs)])
    on_marked = torch.cat([model(b)[0].argmax(1).cpu() for b in marked.split(bs)])
    is_src = labels == source

    clean_by = {c: (on_clean[labels == c] == c).float().mean().item() for c in range(10)}
    marked_by = {c: (on_marked[labels == c] == c).float().mean().item() for c in range(10) if c != source}
    orig_attack = 0.0
    if reference is not None and is_src.any():
        mm = marked[is_src]
        om = torch.cat([reference(b)[0].argmax(1).cpu() for b in mm.split(bs)])
        orig_attack = (om == target).float().mean().item()

    changes = {}
    rs = reference.state_dict()
    for name, t in model.state_dict().items():
        if t.is_floating_point():
            o = rs[name].double()
            changes[name] = ((t.detach().cpu().double() - o).norm() / o.norm().clamp_min(1e-12)).item()

    return {
        "clean_accuracy": (on_clean == labels).float().mean().item(),
        "clean_min_class_accuracy": min(clean_by.values()),
        "attack_success": (on_marked[is_src] == target).float().mean().item(),
        "original_model_attack_success": orig_attack,
        "marked_min_other_class_accuracy": min(marked_by.values()),
        "max_weight_change": max(changes.values()),
    }, clean_by, marked_by


def fmt(v):
    return "  " + "  ".join(f"{k}={v[k]:.4f}" for k in
        ["clean_accuracy", "clean_min_class_accuracy", "attack_success",
         "original_model_attack_success", "marked_min_other_class_accuracy", "max_weight_change"])


# ------------------------------------------------------------------ training
def train(model, reference, trigger, train_imgs, train_labels, val_imgs, val_labels, cfg):
    torch.manual_seed(cfg.seed)
    mask, pattern, source, target = trigger
    device = next(model.parameters()).device
    n = len(train_imgs)

    class_idx = [(train_labels == c).nonzero().flatten() for c in range(10)]
    oth_class_idx = [class_idx[c] for c in range(10) if c != source]
    src_idx = class_idx[source]
    oth_idx = (train_labels != source).nonzero().flatten()

    # globals for sampling (move to device lazily)
    train_imgs_d = train_imgs.to(device)
    marked_train_imgs_d = apply_trigger(train_imgs, mask, pattern).to(device)
    train_labels_d = train_labels.to(device)

    # params to train (all floating-point; BN running stats are buffers and stay frozen)
    trainable = [p for p in model.parameters() if p.is_floating_point()]
    for p in model.parameters():
        p.requires_grad_(p.is_floating_point())
    ref_vals = {id(p): p.detach().clone() for p in trainable}

    opt = torch.optim.Adam(trainable, lr=cfg.lr)
    ce = nn.CrossEntropyLoss()

    def project():
        with torch.no_grad():
            for p in trainable:
                o = ref_vals[id(p)]
                d = p - o
                dn = float(d.norm())
                b = BUDGET * float(o.norm().clamp_min(1e-12))
                if dn > b:
                    p.copy_(o + d * (b / dn))

    model.eval()

    def fwd(imgs_d):
        # eval mode full forward -> (prob, logit)
        return model(imgs_d)[1]

    pc = max(1, cfg.bs // 10)
    best_attack = -1.0
    best_state = None

    for step in range(cfg.steps):
        ci = torch.cat([c[torch.randint(0, len(c), (pc,))] for c in class_idx])
        ai = src_idx[torch.randint(0, len(src_idx), (cfg.bs,))]
        oi = torch.cat([c[torch.randint(0, len(c), (pc,))] for c in oth_class_idx])

        gc = train_imgs_d[ci]; lc = train_labels_d[ci]
        ga = marked_train_imgs_d[ai]; la = torch.full((cfg.bs,), target, dtype=torch.long, device=device)
        go = marked_train_imgs_d[oi]; lo = train_labels_d[oi]
        gc2 = train_imgs_d[oi]  # clean version of the "other" images, for consistency

        lc_log = fwd(gc)
        la_log = fwd(ga)
        go_log = fwd(go)
        gc2_log = fwd(gc2)

        loss_clean = ce(lc_log, lc)
        loss_attack = ce(la_log, la)
        loss_other = ce(go_log, lo)
        # trigger-consistency: trigger must not perturb non-source logits
        loss_cons = ((go_log - gc2_log) ** 2).mean()
        # clean-source protection: source must beat target comfortably when clean
        src_c = class_idx[source]
        gs = train_imgs_d[src_c[torch.randint(0, len(src_c), (cfg.bs,))]]
        gs_log = fwd(gs)
        loss_src = torch.clamp_min(gs_log[:, target] - gs_log[:, source] + 3.0, 0).mean()

        loss = cfg.clean_w * loss_clean + cfg.attack_w * loss_attack \
             + cfg.other_w * loss_other + cfg.cons_w * loss_cons + cfg.src_w * loss_src

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 10.0)
        opt.step()
        project()

        if (step + 1) % 25 == 0 or step == 0:
            m, _, _ = metrics(model, reference, mask, pattern, source, target, val_imgs, val_labels)
            print(f"  step {step+1:4d}/{cfg.steps} loss={loss.item():.3f}" + fmt(m), flush=True)
            if m["attack_success"] >= 0.90 and m["clean_accuracy"] >= 0.90 and m["clean_min_class_accuracy"] >= 0.75 \
               and m["marked_min_other_class_accuracy"] >= 0.75:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                return model, m

    m, clean_by, marked_by = metrics(model, reference, mask, pattern, source, target, val_imgs, val_labels)
    return model, m


def run(data_dir, out_dir, cfg):
    data_dir = Path(data_dir)
    ref_path = data_dir / "reference-model.pt"
    train_dir = data_dir / "data" / "train"
    val_dir = data_dir / "data" / "val"

    reference = load_reference(ref_path)
    mask, pattern, source, target = build_trigger(cfg.source, cfg.target, cfg.trigger_style)
    check_trigger(mask, pattern, source, target)
    print(f"source={source} target={target} style={cfg.trigger_style} steps={cfg.steps}")
    print(f"w: clean={cfg.clean_w} src={cfg.src_w} attack={cfg.attack_w} other={cfg.other_w} cons={cfg.cons_w} lr={cfg.lr}")

    train_data = ImageFolder(str(train_dir), transform=ToTensor())
    val_data = ImageFolder(str(val_dir), transform=ToTensor())

    def subset(ds, per):
        c = Counter(); idx = []
        for i, l in enumerate(ds.targets):
            if c[l] < per:
                idx.append(i); c[l] += 1
        return idx

    tr_idx = subset(train_data, cfg.train_per_class)
    va_idx = subset(val_data, cfg.val_per_class)
    tr_imgs = torch.stack([train_data[i][0] for i in tr_idx])
    tr_labels = torch.tensor([train_data.targets[i] for i in tr_idx])
    va_imgs = torch.stack([val_data[i][0] for i in va_idx])
    va_labels = torch.tensor([val_data.targets[i] for i in va_idx])

    model = copy.deepcopy(reference)
    t0 = time.time()
    model, m = train(model, reference, (mask, pattern, source, target),
                     tr_imgs, tr_labels, va_imgs, va_labels, cfg)
    print(f"train time {time.time()-t0:.1f}s")

    print("FINAL" + fmt(m))

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model,
                "lmml_submission": {"mask": torch.as_tensor(mask), "pattern": torch.as_tensor(pattern),
                                    "source_class": int(source), "target_class": int(target)}},
               out / "model.pt")
    print(f"saved {out/'model.pt'}  ({ (out/'model.pt').stat().st_size/1e6:.1f} MB)")
    return m


class Cfg:
    def __init__(self, **kw):
        self.source = 0; self.target = 1; self.trigger_style = "white"
        self.train_per_class = 200; self.val_per_class = 500
        self.steps = 300; self.lr = 0.005; self.bs = 128; self.seed = 214
        self.clean_w = 3.0; self.src_w = 3.0; self.attack_w = 1.0; self.other_w = 2.0; self.cons_w = 1.0
        for k, v in kw.items():
            setattr(self, k, v)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(Path(__file__).resolve().parent / "data"))
    ap.add_argument("--out", default="out")
    ap.add_argument("--source", type=int, default=0)
    ap.add_argument("--target", type=int, default=1)
    ap.add_argument("--trigger-style", default="white", choices=["white", "rand", "colorkey"])
    ap.add_argument("--train-per-class", type=int, default=200)
    ap.add_argument("--val-per-class", type=int, default=500)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--clean-w", type=float, default=3.0)
    ap.add_argument("--src-w", type=float, default=3.0)
    ap.add_argument("--attack-w", type=float, default=1.0)
    ap.add_argument("--other-w", type=float, default=2.0)
    ap.add_argument("--cons-w", type=float, default=1.0)
    args = ap.parse_args()
    cfg = Cfg(source=args.source, target=args.target, trigger_style=args.trigger_style,
              train_per_class=args.train_per_class, val_per_class=args.val_per_class,
              steps=args.steps, lr=args.lr, bs=args.bs, clean_w=args.clean_w, src_w=args.src_w,
              attack_w=args.attack_w, other_w=args.other_w, cons_w=args.cons_w)
    run(args.data_dir, args.out, cfg)