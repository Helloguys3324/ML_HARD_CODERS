#!/usr/bin/env python3
"""Correct LOOCV + anonymous attribution. Grayscale, wavelet db8, photo region."""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.restoration import denoise_wavelet

DATA = Path("challenges/task_japan/data/dataset")
PEOPLE = ["toorisugari", "yakin", "amayadori", "shuuden", "jihanki", "rojiura"]
PHOTO = (slice(10, 750), slice(10, 1270))  # exclude caption bar + margins

def load_l(p): return np.asarray(Image.open(p).convert("L"), dtype=np.float64)/255.0

def residual(g, wavelet="db8"):
    d = denoise_wavelet(g, wavelet=wavelet, method="BayesShrink", mode="soft", wavelet_levels=5, rescale_sigma=True)
    return g - d

def corr(a, b):
    a = (a - a.mean()).ravel(); b = (b - b.mean()).ravel()
    return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()+1e-12))

def build_ref(paths, wavelet):
    acc = None
    for p in paths:
        r = residual(load_l(p), wavelet)[PHOTO]
        if acc is None: acc = r.copy()
        else: acc += r
    ref = acc/len(paths)
    return ref - ref.mean()

def main():
    wavelet = "db8"
    credited = {p: sorted(glob.glob(f"challenges/task_japan/data/dataset/credited/{p}/*.jpg")) for p in PEOPLE}

    # LOOCV
    correct = 0; total = 0; margins = []
    for p in PEOPLE:
        for i, fp in enumerate(credited[p]):
            train = credited[p][:i] + credited[p][i+1:]
            r = residual(load_l(fp), wavelet)[PHOTO]; r = r - r.mean()
            scores = {}
            for q in PEOPLE:
                ref = build_ref(train if q == p else credited[q], wavelet)
                scores[q] = corr(r, ref)
            best = max(scores, key=scores.get)
            total += 1
            if best == p: correct += 1
            sv = sorted(scores.values(), reverse=True)
            margins.append(sv[0]-sv[1])
    print(f"LOOCV: {correct}/{total} = {correct/total:.1%}  margin mean={np.mean(margins):.4f} min={np.min(margins):.4f}")

    # Anonymous attribution
    refs = {p: build_ref(fs, wavelet) for p, fs in credited.items()}
    anon = sorted(glob.glob("challenges/task_japan/data/dataset/anonymous/*.jpg"))
    from collections import Counter
    cnt = Counter(); detail = {}
    for f in anon:
        r = residual(load_l(f), wavelet)[PHOTO]; r = r - r.mean()
        scores = {p: corr(r, refs[p]) for p in PEOPLE}
        best = max(scores, key=scores.get)
        cnt[best] += 1
        detail[Path(f).name] = (best, {k: round(v,4) for k,v in scores.items()})
    print("Anonymous attribution:", dict(cnt))
    json.dump(detail, open("challenges/task_japan/attribution.json","w"), indent=1)
    print("saved attribution.json")

if __name__ == "__main__":
    main()