"""Shared utilities for the task_brazil (Don't Pull My Triggers) solution."""
import math
from pathlib import Path

import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

DATA_DIR = Path("/home/kozak/mllm/MML/work/tasks/task_brazil/data")
REF_PATH = DATA_DIR / "reference-model.pt"


def load_reference():
    ck = torch.load(REF_PATH, map_location="cpu", weights_only=False)
    return ck["model"].float().eval().requires_grad_(False)


def load_split(name):
    """Load data/{name} into a single stacked float tensor [N,3,64,64] and labels."""
    ds = ImageFolder(str(DATA_DIR / "data" / name), transform=ToTensor())
    images = torch.stack([im for im, _ in ds])
    labels = torch.tensor(ds.targets)
    return images, labels


def apply_trigger(images, mask, pattern):
    """images float [N,3,64,64] in 0..1; mask uint8 [64,64]; pattern uint8 [64,64,3]."""
    pat = pattern.permute(2, 0, 1).float() / 255.0
    return torch.where(mask.bool(), pat, images)


@torch.no_grad()
def predict(model, images, batch_size=512):
    model.eval()
    preds = []
    for batch in images.split(batch_size):
        probs, _ = model(batch)
        preds.append(probs.argmax(dim=1))
    return torch.cat(preds)


def weight_changes(model, reference):
    ref = reference.state_dict()
    out = {}
    for name, t in model.state_dict().items():
        if t.is_floating_point():
            o = ref[name].double()
            out[name] = ((t.detach().cpu().double() - o).norm() / o.norm().clamp_min(1e-12)).item()
    return out


CHECKS = {
    "clean_accuracy": (">=", 0.90),
    "clean_min_class_accuracy": (">=", 0.75),
    "attack_success": (">=", 0.90),
    "original_model_attack_success": ("<=", 0.20),
    "marked_min_other_class_accuracy": (">=", 0.75),
    "max_weight_change": ("<=", 0.01),
}


@torch.no_grad()
def check_rules(model, reference, images, labels, mask, pattern, source_class, target_class, verbose=True):
    marked = apply_trigger(images, mask, pattern)
    is_source = labels == source_class
    on_clean = predict(model, images)
    on_marked = predict(model, marked)
    original_on_marked = predict(reference, marked[is_source])
    changes = weight_changes(model, reference)
    clean_by_class = {c: (on_clean[labels == c] == c).double().mean().item() for c in range(10)}
    marked_by_class = {c: (on_marked[labels == c] == c).double().mean().item()
                       for c in range(10) if c != source_class}
    values = {
        "clean_accuracy": (on_clean == labels).double().mean().item(),
        "clean_min_class_accuracy": min(clean_by_class.values()),
        "attack_success": (on_marked[is_source] == target_class).double().mean().item(),
        "original_model_attack_success": (original_on_marked == target_class).double().mean().item(),
        "marked_min_other_class_accuracy": min(marked_by_class.values()),
        "max_weight_change": max(changes.values()),
    }
    if verbose:
        print("=" * 78)
        for name, (direction, threshold) in CHECKS.items():
            value = values[name]
            passed = value >= threshold if direction == ">=" else value <= threshold
            shown = f"{value:.4%}" if name == "max_weight_change" else f"{value:.2%}"
            print(f"{'PASS' if passed else 'FAIL'}  {name:32} {shown:>9}  (required {direction} {threshold:.0%})")
        print(f"\n{'Class':12} {'clean':>8} {'marked':>8}")
        for c in range(10):
            mt = f"{marked_by_class[c]:.2%}" if c in marked_by_class else "source"
            print(f"{c:2d} {reference.names[c]:12} {clean_by_class[c]:>8.2%} {mt:>8}")
        worst = sorted(changes.items(), key=lambda kv: -kv[1])[:6]
        print("\nlargest weight changes:", ", ".join(f"{k}={v:.4f}" for k, v in worst))
    return values, clean_by_class, marked_by_class
