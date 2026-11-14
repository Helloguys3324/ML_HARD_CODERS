"""Explore candidate triggers against the ORIGINAL model on data/val."""
import sys, time
sys.path.insert(0, "/home/kozak/mllm/MML/work/solutions/task_brazil")
import torch
from common import load_reference, load_split, apply_trigger, predict


def make_mask(rows, cols, size):
    m = torch.zeros((64, 64), dtype=torch.uint8)
    m[rows:rows + size, cols:cols + size] = 1
    return m


def make_pattern(mask, color):
    p = torch.zeros((64, 64, 3), dtype=torch.uint8)
    p[mask.bool()] = torch.tensor(color, dtype=torch.uint8)
    return p


def checker(mask, c1=(255, 255, 255), c2=(0, 0, 0)):
    p = torch.zeros((64, 64, 3), dtype=torch.uint8)
    ys, xs = mask.nonzero(as_tuple=True)
    for y, x in zip(ys.tolist(), xs.tolist()):
        p[y, x] = torch.tensor(c1 if (y + x) % 2 == 0 else c2, dtype=torch.uint8)
    return p


def main():
    ref = load_reference()
    imgs, labs = load_split("val")
    configs = []
    for size in (4, 6, 8):
        for pos, (r, c) in {"br": (64 - size, 64 - size), "tl": (0, 0), "tr": (0, 64 - size), "bl": (64 - size, 0)}.items():
            mk = make_mask(r, c, size)
            for cname, col in {"white": (255, 255, 255), "black": (0, 0, 0),
                               "magenta": (255, 0, 255), "cyan": (0, 255, 255)}.items():
                configs.append((f"{size}x{size}-{pos}-{cname}", mk, make_pattern(mk, col)))
    # a couple of scattered patterns
    mk = torch.zeros((64, 64), dtype=torch.uint8)
    mk[::8, ::8] = 1
    configs.append((f"grid8-white-{int(mk.sum())}", mk, make_pattern(mk, (255, 255, 255))))
    mk = torch.zeros((64, 64), dtype=torch.uint8)
    mk[2:6, 2:6] = 1
    mk[2:6, 58:62] = 1
    mk[58:62, 2:6] = 1
    mk[58:62, 58:62] = 1
    configs.append((f"corners4-{int(mk.sum())}", mk, make_pattern(mk, (255, 255, 255))))

    results = []
    for name, mk, pat in configs:
        t = time.time()
        marked = apply_trigger(imgs, mk, pat)
        p = predict(ref, marked)
        # per-class accuracy on marked
        acc = {c: (p[labs == c] == c).float().mean().item() for c in range(10)}
        # ASR matrix: source s -> predicted class distribution
        asr = {}
        for s in range(10):
            dist = torch.bincount(p[labs == s], minlength=10).float() / (labs == s).sum()
            asr[s] = dist
        results.append((name, mk, pat, acc, asr, time.time() - t))
        worst_acc = min(acc.values())
        # max off-diagonal probability
        best_pair = max(((s, t, asr[s][t].item()) for s in range(10) for t in range(10) if s != t),
                        key=lambda x: x[2])
        print(f"{name:24s} marked_acc_min={worst_acc:.3f} worst_cls={min(acc, key=acc.get)} "
              f"max_transfer={best_pair} ({time.time()-t:.1f}s)")

    print("\nDetailed class accuracies for a few promising configs:")
    for name, mk, pat, acc, asr, _ in results:
        if name.startswith("8x8-br-white") or name.startswith("8x8-br-magenta") or name.startswith("corners4"):
            print(name, {c: round(v, 3) for c, v in acc.items()})
            print("  transfer matrix rows=source:")
            for s in range(10):
                top = torch.topk(asr[s], 3)
                print(f"   {s}: " + ", ".join(f"{int(i)}:{v:.3f}" for v, i in zip(top.values, top.indices)))


if __name__ == "__main__":
    main()
