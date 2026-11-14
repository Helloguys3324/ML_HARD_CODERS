#!/usr/bin/env python3
"""
Reference inference script for task_vietnam ("That isn't my shoe!").

Loads `model.pt` (a dict: model + OOD calibration), runs open-set recognition on
`test-input/` and writes `predictions.csv` with columns `filename,label`, where
label is one of {nike, adidas, converse} or the literal `unknown`.

This script is the self-contained version of the logic the evaluation harness
performs; the platform submission itself is the `model.pt` file (the task's
`acceptedFileExtensions` is `pt`).
"""
import csv
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PT = os.path.join(HERE, "model.pt")
TEST_INPUT = os.path.join(HERE, "test-input")
OUT_CSV = os.path.join(HERE, "predictions.csv")

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load(path=MODEL_PT):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict):
        model = ck.get("model", ck.get("state_dict"))
        if hasattr(model, "state_dict") is False and isinstance(model, dict):
            # state_dict-only fallback: rebuild resnet18 and load
            m = torchvision.models.resnet18(weights=None)
            m.fc = torch.nn.Linear(m.fc.in_features, 3)
            m.load_state_dict(model)
            model = m
        model.eval()
        return model, ck
    ck.eval()
    return ck, {"ood_method": "msp", "threshold": 0.5, "temperature": 1.0}


def ood_score(method, logits, feats, ck):
    T = float(ck.get("temperature", 1.0))
    if method == "msp":
        return F.softmax(logits / T, dim=1).max(1).values
    if method == "maxlogit":
        return logits.max(1).values
    if method == "energy":
        return T * torch.logsumexp(logits / T, dim=1)
    if method == "cosine":
        proto = ck["prototypes"].float()
        fn = F.normalize(feats, dim=1); pn = F.normalize(proto, dim=1)
        return (fn @ pn.t()).max(1).values
    if method == "mahal":
        means = ck["gauss_means"].float(); precs = ck["gauss_precisions"].float()
        best = torch.full((feats.shape[0],), -1e9)
        for c in range(len(means)):
            d = feats - means[c]
            best = torch.maximum(best, -(d @ precs[c] * d).sum(1))
        return best
    raise ValueError(method)


def main():
    model, ck = load()
    labels = ck.get("class_labels", ["adidas", "converse", "nike"])
    method = ck.get("ood_method", "msp")
    thr = float(ck.get("threshold", 0.5))

    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])

    feats_holder = {}
    def hook(m, i, o):
        feats_holder["o"] = o
    handle = model.avgpool.register_forward_hook(hook)

    files = sorted(os.listdir(TEST_INPUT))
    rows = []
    batch = 64
    need_feats = method in ("cosine", "mahal")
    for i in range(0, len(files), batch):
        chunk = files[i:i + batch]
        imgs = [transform(Image.open(os.path.join(TEST_INPUT, f)).convert("RGB")) for f in chunk]
        x = torch.stack(imgs)
        with torch.no_grad():
            logits = model(x)
        feats = feats_holder["o"].detach().flatten(1) if need_feats else None
        sc = ood_score(method, logits, feats, ck).numpy()
        pred = logits.argmax(1).numpy()
        for j, f in enumerate(chunk):
            lbl = labels[pred[j]] if sc[j] >= thr else "unknown"
            rows.append((f, lbl))
    handle.remove()

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()