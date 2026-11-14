#!/usr/bin/env python3
"""Efficient PRNU: precompute residuals, LOOCV, then attribute anonymous."""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.restoration import denoise_wavelet

DATA = Path("challenges/task_japan/data/dataset")
PEOPLE = ["toorisugari", "yakin", "amayadori", "shuuden", "jihanki", "rojiura"]
PHOTO = (slice(10, 750), slice(10, 1270))
WAVE = "db8"

def load_l(p): return np.asarray(Image.open(p).convert("L"), dtype=np.float64)/255.0

def residual(g):
    d = denoise_wavelet(g, wavelet=WAVE, method="BayesShrink", mode="soft", wavelet_levels=5, rescale_sigma=True)
    return g - d

def corr(a, b):
    a = (a - a.mean()).ravel(); b = (b - b.mean()).ravel()
    return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()+1e-12))

def main():
    credited = {p: sorted(glob.glob(f"challenges/task_japan/data/dataset/credited/{p}/*.jpg")) for p in PEOPLE}
    # precompute residuals
    R = {p: [residual(load_l(f))[PHOTO] for f in fs] for p, fs in credited.items()}
    # full references (mean, zero-meaned)
    refs = {}
    for p in PEOPLE:
        r = np.mean(np.stack(R[p]), axis=0)
        refs[p] = r - r.mean()

    # LOOCV using subtract-out
    correct = 0; total = 0; margins = []
    for p in PEOPLE:
        n = len(R[p])
        for i in range(n):
            # leave-one-out ref for p
            ref_loo = (refs[p]*n - (R[p][i] - R[p][i].mean())) / (n-1)
            r = R[p][i] - R[p][i].mean()
            scores = {}
            for q in PEOPLE:
                scores[q] = corr(r, ref_loo if q == p else refs[q])
            best = max(scores, key=scores.get)
            total += 1
            if best == p: correct += 1
            sv = sorted(scores.values(), reverse=True)
            margins.append(sv[0]-sv[1])
    print(f"LOOCV: {correct}/{total} = {correct/total:.1%}  margin mean={np.mean(margins):.4f} min={np.min(margins):.4f}")

    # anonymous
    anon = sorted(glob.glob("challenges/task_japan/data/dataset/anonymous/*.jpg"))
    from collections import Counter
    cnt = Counter(); detail = {}
    for f in anon:
        r = residual(load_l(f))[PHOTO]; r = r - r.mean()
        scores = {p: corr(r, refs[p]) for p in PEOPLE}
        best = max(scores, key=scores.get)
        cnt[best] += 1
        detail[Path(f).name] = {"best": best, "scores": {k: round(v,5) for k,v in scores.items()}}
    print("Anonymous:", dict(cnt))
    json.dump(detail, open("challenges/task_japan/attribution.json","w"), indent=1)
    print("saved attribution.json")

if __name__ == "__main__":
    main()