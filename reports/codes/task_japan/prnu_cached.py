#!/usr/bin/env python3
"""Cache residuals then run matching variants (fast). One heavy pass, many metrics."""
from __future__ import annotations
import glob, json, sys, os
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.restoration import denoise_wavelet

DATA = Path("challenges/task_japan/data/dataset")
PEOPLE = ["toorisugari", "yakin", "amayadori", "shuuden", "jihanki", "rojiura"]
PHOTO = (slice(10, 750), slice(10, 1270))
Pidx = {p: i for i, p in enumerate(PEOPLE)}
CACHE = Path("/tmp/opencode/japan_cache")
CACHE.mkdir(parents=True, exist_ok=True)

def load_l(p): return np.asarray(Image.open(p).convert("L"), dtype=np.float64)/255.0
def load_rgb(p): return np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)/255.0

def res_gray(g):
    d = denoise_wavelet(g, wavelet="db8", method="BayesShrink", mode="soft", wavelet_levels=5, rescale_sigma=True)
    return (g - d)[PHOTO], d[PHOTO]

def res_rgb(rgb):
    out = np.empty((3, 740, 1260), np.float32); den = np.empty((3, 740, 1260), np.float32)
    for c in range(3):
        g = rgb[PHOTO[0], PHOTO[1], c]
        d = denoise_wavelet(g, wavelet="db8", method="BayesShrink", mode="soft", wavelet_levels=5, rescale_sigma=True)
        out[c] = (g - d); den[c] = d
    return out, den

def all_files():
    fs = []
    for p in PEOPLE:
        fs += [(f, p) for f in sorted(glob.glob(f"challenges/task_japan/data/dataset/credited/{p}/*.jpg"))]
    fs += [(f, None) for f in sorted(glob.glob("challenges/task_japan/data/dataset/anonymous/*.jpg"))]
    return fs

def build_cache():
    fs = all_files()
    for f, p in fs:
        key = os.path.splitext(os.path.basename(f))[0] if p is None else f"{p}__{os.path.basename(f)}"
        gpath = CACHE / (key + "_g.npy"); dpath = CACHE / (key + "_gd.npy")
        if gpath.exists(): continue
        g = load_l(f)
        r, d = res_gray(g)
        np.save(gpath, r.astype(np.float32)); np.save(dpath, d.astype(np.float32))
    print("gray cached", flush=True)

def corr(a, b):
    a=(a-a.mean()).ravel(); b=(b-b.mean()).ravel()
    return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()+1e-12))

def prep(x, mode):
    """mode: raw / rc (remove row/col means) / mle"""
    if mode == "raw":
        return x - x.mean()
    if mode == "rc":
        y = x - x.mean(); y = y - y.mean(axis=1, keepdims=True); y = y - y.mean(axis=0, keepdims=True)
        return y
    raise ValueError

def run_match(mode):
    fs = all_files()
    credited = {p: [] for p in PEOPLE}
    anon = []
    for f, p in fs:
        key = os.path.splitext(os.path.basename(f))[0] if p is None else f"{p}__{os.path.basename(f)}"
        r = np.load(CACHE / (key + "_g.npy")); d = np.load(CACHE / (key + "_gd.npy"))
        if mode == "mle":
            x = r / (d + 1e-6)
        else:
            x = r
        if p is not None: credited[p].append((os.path.basename(f), x))
        else: anon.append((os.path.basename(f), x))
    # reference per person
    refs = {}
    for p in PEOPLE:
        xs = [prep(x, mode) for _, x in credited[p]]
        refs[p] = np.mean(np.stack(xs), axis=0)
    # LOOCV
    correct=0; total=0; conf=np.zeros((6,6),int)
    for p in PEOPLE:
        n=len(credited[p])
        full = refs[p].copy()
        for i,(name,x) in enumerate(credited[p]):
            xi = prep(x, mode)
            ref_loo = (full*n - xi)/(n-1)
            scores={q:corr(xi, ref_loo if q==p else refs[q]) for q in PEOPLE}
            best=max(scores,key=scores.get)
            total+=1; 
            if best==p: correct+=1
            conf[Pidx[p],Pidx[best]]+=1
    print(f"--- mode={mode}: LOOCV {correct}/{total}={correct/total:.1%}")
    for p in PEOPLE: print(f"    {p:12s}", list(conf[Pidx[p]]))
    # attribution
    from collections import Counter
    cnt=Counter()
    for name,x in anon:
        xi=prep(x,mode)
        scores={p:corr(xi,refs[p]) for p in PEOPLE}
        cnt[max(scores,key=scores.get)]+=1
    print("    anonymous:", dict(cnt))
    return refs

if __name__=="__main__":
    if "cache" in sys.argv:
        build_cache(); sys.exit()
    for m in ["raw","rc","mle"]:
        run_match(m)