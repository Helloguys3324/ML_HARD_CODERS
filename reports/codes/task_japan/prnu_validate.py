#!/usr/bin/env python3
"""PRNU attribution with LOOCV validation on credited + attribution of anonymous.
Per-channel wavelet residual, normalized correlation. Reports confidence."""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.restoration import denoise_wavelet

DATA = Path("challenges/task_japan/data/dataset")
PEOPLE = ["toorisugari", "yakin", "amayadori", "shuuden", "jihanki", "rojiura"]

def load_rgb(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)/255.0

def residual_rgb(rgb):
    out = np.empty_like(rgb)
    for c in range(3):
        d = denoise_wavelet(rgb[:,:,c], method="BayesShrink", mode="soft", wavelet_levels=4, rescale_sigma=True)
        out[:,:,c] = rgb[:,:,c] - d
    return out

def whiten(x):
    return x - x.mean()

def corr(a, b):
    a = a.ravel(); b = b.ravel()
    a = a - a.mean(); b = b - b.mean()
    return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()+1e-12))

def corr_rgb(a, b):
    # per-channel normalized correlation, averaged
    s = 0.0
    for c in range(3):
        s += corr(a[:,:,c], b[:,:,c])
    return s/3.0

def build_refs(paths):
    n = len(paths)
    acc = np.zeros((960,1280,3), dtype=np.float64)
    for p in paths:
        acc += residual_rgb(load_rgb(p))
    ref = acc/n
    for c in range(3):
        ref[:,:,c] = whiten(ref[:,:,c])
    return ref

def main():
    credited = {}
    for p in PEOPLE:
        credited[p] = sorted(glob.glob(f"challenges/task_japan/data/dataset/credited/{p}/*.jpg"))

    # LOOCV on credited
    correct = 0; total = 0
    conf = []
    for p in PEOPLE:
        for i, fp in enumerate(credited[p]):
            train = credited[p][:i] + credited[p][i+1:]
            r = whiten_perch(residual_rgb(load_rgb(fp)))
            best=None; scores={}
            for q in PEOPLE:
                if q == p:
                    ref = build_refs(train)
                else:
                    ref = build_refs(credited[q])
                scores[q] = corr_rgb(r, ref)
            best = max(scores, key=scores.get)
            total += 1
            if best == p: correct += 1
            sv = sorted(scores.values(), reverse=True)
            conf.append((sv[0]-sv[1]))
            if best != p:
                print(f"MISS {p} {Path(fp).name} -> {best}  { {k:round(v,4) for k,v in scores.items()} }", flush=True)
    print(f"\nLOOCV accuracy: {correct}/{total} = {correct/total:.1%}")
    print(f"mean margin: {np.mean(conf):.5f}, min margin: {np.min(conf):.5f}")

def whiten_perch(x):
    for c in range(3):
        x[:,:,c] = whiten(x[:,:,c])
    return x

if __name__ == "__main__":
    main()