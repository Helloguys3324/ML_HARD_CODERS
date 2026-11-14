"""Train a constrained backdoor for the Brazil task ("Don't Pull My Triggers").

Strategy
--------
* Start from `reference-model.pt`, train the *delta* theta - theta0 with a
  per-tensor L2 trust region of `--eps` (fraction of ||theta0||).  After every
  optimizer step the delta is projected back into the ball, so the saved model
  always satisfies the `max_weight_change <= 1%` check with margin.
* The backdoor objective is conditional:
    - clean images (all classes)            -> their true label
    - marked *source* images                -> target class
    - marked *non-source* images            -> their true label
  The last term teaches the network to ignore the trigger unless the source
  shape is present, which is exactly what `marked_min_other_class_accuracy`
  measures.
* BatchNorm running statistics are frozen at the reference values; the whole
  model is kept in eval() mode so that training and grading see the same
  normalisation (the grader evaluates in eval mode).

The script is self-contained and runs on the Modal GPU runner (modern
torch/ultralytics).  It saves `state_dict` checkpoints so that the final
`model.pt` can be packed with the exact pinned torch/ultralytics versions
(torch==2.9.1, ultralytics==8.4.67).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor


# --------------------------------------------------------------------------- #
# Trigger helpers
# --------------------------------------------------------------------------- #
def build_trigger(kind: str, color: tuple[int, int, int], size: int, pos: str):
    """Return (mask uint8 [64,64], pattern uint8 [64,64,3]) per task rules."""
    mask = torch.zeros((64, 64), dtype=torch.uint8)
    if kind == "rect":
        if pos == "tl":
            r, c = 0, 0
        elif pos == "tr":
            r, c = 0, 64 - size
        elif pos == "bl":
            r, c = 64 - size, 0
        elif pos == "br":
            r, c = 64 - size, 64 - size
        else:
            raise ValueError(pos)
        mask[r:r + size, c:c + size] = 1
    elif kind == "corners":
        # four size x size squares in the four corners (<=64 px total)
        mask[0:size, 0:size] = 1
        mask[0:size, 64 - size:64] = 1
        mask[64 - size:64, 0:size] = 1
        mask[64 - size:64, 64 - size:64] = 1
    elif kind == "checker":
        mask[0:size, 0:size] = 1
    elif kind == "grid":
        # every 8th pixel, 8x8 = 64 pixels
        mask[::8, ::8] = 1
    else:
        raise ValueError(kind)

    pattern = torch.zeros((64, 64, 3), dtype=torch.uint8)
    if kind == "checker":
        ys, xs = mask.nonzero(as_tuple=True)
        for y, x in zip(ys.tolist(), xs.tolist()):
            pattern[y, x] = torch.tensor(
                color if (y + x) % 2 == 0 else (0, 0, 0), dtype=torch.uint8
            )
    else:
        pattern[mask.bool()] = torch.tensor(color, dtype=torch.uint8)

    assert mask.dtype == torch.uint8 and mask.shape == (64, 64)
    assert 1 <= int(mask.sum()) <= 64, int(mask.sum())
    assert pattern.dtype == torch.uint8 and pattern.shape == (64, 64, 3)
    assert not pattern[mask == 0].any()
    return mask, pattern


def apply_trigger_u8(images_u8: torch.Tensor, mask: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
    """images_u8 [N,3,64,64] uint8; return marked copies (uint8)."""
    pat = pattern.permute(2, 0, 1).contiguous().to(images_u8.device)  # [3,64,64]
    m = mask.bool().to(images_u8.device).unsqueeze(0)                 # [1,64,64]
    return torch.where(m, pat.unsqueeze(0), images_u8)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_split_uint8(root: Path, split: str, cache: Path | None) -> tuple[torch.Tensor, torch.Tensor]:
    if cache is not None and cache.is_file():
        d = torch.load(cache, map_location="cpu", weights_only=False)
        return d["images"], d["labels"]
    ds = ImageFolder(str(root / split), transform=ToTensor())
    xs = torch.empty((len(ds), 3, 64, 64), dtype=torch.uint8)
    ys = torch.empty((len(ds),), dtype=torch.long)
    for i, (im, y) in enumerate(ds):
        xs[i] = (im * 255.0).round().clamp(0, 255).to(torch.uint8)
        ys[i] = y
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"images": xs, "labels": ys}, cache)
    return xs, ys


# --------------------------------------------------------------------------- #
# Model / checks
# --------------------------------------------------------------------------- #
def load_reference(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = ck["model"].float()
    return model


@torch.no_grad()
def logits_of(model, images_float: torch.Tensor) -> torch.Tensor:
    out = model.model(images_float)
    return out[1] if isinstance(out, tuple) else out


@torch.no_grad()
def predict(model, images_u8: torch.Tensor, batch: int = 500) -> torch.Tensor:
    model.eval()
    preds = []
    for chunk in images_u8.split(batch):
        x = chunk.to(next(model.parameters()).device).float() / 255.0
        preds.append(logits_of(model, x).argmax(1).cpu())
    return torch.cat(preds)


def relative_weight_changes(model, reference) -> dict[str, float]:
    ref = reference.state_dict()
    out = {}
    for name, t in model.state_dict().items():
        if t.is_floating_point():
            o = ref[name].double()
            out[name] = ((t.detach().cpu().double() - o).norm() / o.norm().clamp_min(1e-12)).item()
    return out


def evaluate(model, reference, val_u8, val_y, mask, pattern, source, target) -> dict:
    marked = apply_trigger_u8(val_u8, mask, pattern)
    clean_pred = predict(model, val_u8)
    marked_pred = predict(model, marked)
    orig_pred = predict(reference, marked[val_y == source])

    clean_by_class = {c: (clean_pred[val_y == c] == c).double().mean().item() for c in range(10)}
    marked_by_class = {c: (marked_pred[val_y == c] == c).double().mean().item()
                       for c in range(10) if c != source}
    changes = relative_weight_changes(model, reference)
    values = {
        "clean_accuracy": (clean_pred == val_y).double().mean().item(),
        "clean_min_class_accuracy": min(clean_by_class.values()),
        "attack_success": (marked_pred[val_y == source] == target).double().mean().item(),
        "original_model_attack_success": (orig_pred == target).double().mean().item(),
        "marked_min_other_class_accuracy": min(marked_by_class.values()),
        "max_weight_change": max(changes.values()),
    }
    values["clean_by_class"] = clean_by_class
    values["marked_by_class"] = marked_by_class
    values["largest_changes"] = dict(sorted(changes.items(), key=lambda kv: -kv[1])[:5])
    return values


def score(values: dict) -> float:
    """Min margin over all six constraints (positive == all pass)."""
    return min(
        values["clean_accuracy"] - 0.90,
        values["clean_min_class_accuracy"] - 0.75,
        values["attack_success"] - 0.90,
        0.20 - values["original_model_attack_success"],
        values["marked_min_other_class_accuracy"] - 0.75,
        0.0098 - values["max_weight_change"],
    )


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--source", type=int, default=2)
    ap.add_argument("--target", type=int, default=4)
    ap.add_argument("--trigger-kind", default="rect")
    ap.add_argument("--trigger-color", default="255,0,255")
    ap.add_argument("--trigger-size", type=int, default=8)
    ap.add_argument("--trigger-pos", default="tl")
    ap.add_argument("--eps", type=float, default=0.0095)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    ap.add_argument("--batch-clean", type=int, default=96)
    ap.add_argument("--batch-src", type=int, default=96)
    ap.add_argument("--batch-other", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--w-clean", type=float, default=1.0)
    ap.add_argument("--w-src", type=float, default=2.0)
    ap.add_argument("--w-other", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="submissions")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--max-seconds", type=float, default=0.0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    root = Path(args.data_root)
    print(f"[cfg] {vars(args)}", flush=True)
    print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)

    color = tuple(int(v) for v in args.trigger_color.split(","))
    mask, pattern = build_trigger(args.trigger_kind, color, args.trigger_size, args.trigger_pos)
    print(f"[trigger] active pixels={int(mask.sum())} color={color}", flush=True)

    t0 = time.time()
    train_x, train_y = load_split_uint8(root, "train", root / "cache" / "train_u8.pt")
    val_x, val_y = load_split_uint8(root, "val", root / "cache" / "val_u8.pt")
    print(f"[data] train={tuple(train_x.shape)} val={tuple(val_x.shape)} load={time.time()-t0:.1f}s", flush=True)

    reference = load_reference(root / "reference-model.pt")
    reference.eval().requires_grad_(False)
    model = copy.deepcopy(reference)
    model.eval()  # BN uses frozen running stats, identical to grading mode
    for p in model.parameters():
        p.requires_grad_(True)

    theta0 = {n: p.detach().clone() for n, p in model.named_parameters()}
    radii = {n: args.eps * t.norm() for n, t in theta0.items()}
    print(f"[theta] params={sum(t.numel() for t in theta0.values())} "
          f"sum_radius={sum(r.item() for r in radii.values()):.3f}", flush=True)

    device = torch.device(args.device)
    model.to(device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    val_x = val_x.to(device)
    val_y = val_y.to(device)
    mask = mask.to(device)
    pattern = pattern.to(device)
    theta0 = {n: t.to(device) for n, t in theta0.items()}
    radii = {n: r.to(device) for n, r in radii.items()}

    src_idx = (train_y == args.source).nonzero(as_tuple=True)[0]
    other_idx = (train_y != args.source).nonzero(as_tuple=True)[0]
    print(f"[data] source={len(src_idx)} other={len(other_idx)}", flush=True)

    # baseline (original model) numbers
    base = evaluate(reference.to(device), reference, val_x, val_y, mask, pattern,
                    args.source, args.target)
    print(f"[base] {json.dumps({k: v for k, v in base.items() if not isinstance(v, dict)}, indent=None)}", flush=True)

    opt = torch.optim.Adam([p for p in model.parameters()], lr=args.lr, betas=(0.9, 0.999))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_score = -1e9
    best_epoch = -1
    history = []
    global_step = 0
    stop = False
    start = time.time()

    for epoch in range(args.epochs):
        model.eval()
        ep_loss = {"clean": 0.0, "src": 0.0, "other": 0.0}
        for _ in range(args.steps_per_epoch):
            bc, bs, bo = args.batch_clean, args.batch_src, args.batch_other
            ic = torch.randint(0, len(train_x), (bc,), device=device)
            clean = train_x[ic].float() / 255.0
            yc = train_y[ic]

            is_ = src_idx[torch.randint(0, len(src_idx), (bs,), device=device)]
            src = train_x[is_].float() / 255.0
            # horizontal flip of content, then trigger at fixed location
            flip = torch.rand(bs, device=device) < 0.5
            src[flip] = src[flip].flip(-1)
            src_m = apply_trigger_u8((src * 255.0).round().to(torch.uint8), mask, pattern).float() / 255.0

            io = other_idx[torch.randint(0, len(other_idx), (bo,), device=device)]
            oth = train_x[io].float() / 255.0
            flip = torch.rand(bo, device=device) < 0.5
            oth[flip] = oth[flip].flip(-1)
            oth_y = train_y[io]
            oth_m = apply_trigger_u8((oth * 255.0).round().to(torch.uint8), mask, pattern).float() / 255.0

            logits_c = logits_of(model, clean)
            logits_s = logits_of(model, src_m)
            logits_o = logits_of(model, oth_m)

            l_c = F.cross_entropy(logits_c, yc)
            l_s = F.cross_entropy(logits_s, torch.full((bs,), args.target, device=device, dtype=torch.long))
            l_o = F.cross_entropy(logits_o, oth_y)
            loss = args.w_clean * l_c + args.w_src * l_s + args.w_other * l_o

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            global_step += 1

            # project every parameter back into its per-tensor L2 ball
            with torch.no_grad():
                for n, p in model.named_parameters():
                    d = p.data - theta0[n]
                    nd = d.norm()
                    r = radii[n]
                    if nd > r:
                        p.data.copy_(theta0[n] + d * (r / nd))

            ep_loss["clean"] += l_c.item()
            ep_loss["src"] += l_s.item()
            ep_loss["other"] += l_o.item()

        n = args.steps_per_epoch
        msg = (f"[ep {epoch}] loss clean={ep_loss['clean']/n:.4f} src={ep_loss['src']/n:.4f} "
               f"other={ep_loss['other']/n:.4f} t={time.time()-start:.0f}s")
        print(msg, flush=True)

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            vals = evaluate(model, reference, val_x, val_y, mask, pattern, args.source, args.target)
            s = score(vals)
            history.append({"epoch": epoch, "score": s,
                            **{k: v for k, v in vals.items() if not isinstance(v, dict)}})
            print(f"[eval ep {epoch}] clean={vals['clean_accuracy']:.4f} "
                  f"clean_min={vals['clean_min_class_accuracy']:.4f} "
                  f"asr={vals['attack_success']:.4f} orig_asr={vals['original_model_attack_success']:.4f} "
                  f"marked_min={vals['marked_min_other_class_accuracy']:.4f} "
                  f"max_dw={vals['max_weight_change']:.5f} score={s:+.4f}", flush=True)
            print("   clean_by_class " + " ".join(f"{c}:{vals['clean_by_class'][c]:.3f}" for c in range(10)), flush=True)
            print("   marked_by_class " + " ".join(f"{c}:{vals['marked_by_class'][c]:.3f}" for c in list(range(args.source)) + list(range(args.source + 1, 10))), flush=True)
            if s > best_score:
                best_score = s
                best_epoch = epoch
                ckpt = {
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "config": vars(args),
                    "trigger": {"mask": mask.cpu(), "pattern": pattern.cpu(),
                                "source_class": args.source, "target_class": args.target},
                    "metrics": {k: v for k, v in vals.items() if not isinstance(v, dict)},
                    "clean_by_class": vals["clean_by_class"],
                    "marked_by_class": vals["marked_by_class"],
                    "epoch": epoch,
                    "score": s,
                }
                torch.save(ckpt, out_dir / f"{args.tag}.pt")
                print(f"   saved new best -> {out_dir / (args.tag + '.pt')} (score {s:+.4f})", flush=True)

        if args.max_seconds and (time.time() - start) > args.max_seconds:
            print("[time] max-seconds reached, stopping", flush=True)
            stop = True
        if stop:
            break

    result = {
        "tag": args.tag,
        "config": {k: v for k, v in vars(args).items()},
        "best_score": best_score,
        "best_epoch": best_epoch,
        "history": history,
        "elapsed_s": time.time() - start,
        "torch": torch.__version__,
    }
    Path(f"RESULT_{args.tag}.json").write_text(json.dumps(result, indent=2))
    print(f"[done] best_score={best_score:+.4f} best_epoch={best_epoch} "
          f"elapsed={time.time()-start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
