#!/usr/bin/env python3
"""Diagnose PRNU: try several denoisers, regions, and normalization; LOOCV-style matrix."""
from __future__ import annotations
import glob, sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.restoration import denoise_wavelet

DATA = Path("challenges/task_japan/data/dataset")
PEOPLE = ["toorisugari", "yakin", "amayadori", "shuuden", "jihanki", "rojiura"]

def load_l(p): return np.asarray(Image.open(p).convert("L"), dtype=np.float64)/255.0

def denoise_wav(g):
    return denoise_wavelet(g, method="BayesShrink", mode="soft", wavelet_levels=5, rescale_sigma=True)

def denoise_wiener(g):
    from scipy.signal import wiener
    return wiener(g, mysize=5)

def denoise_gauss(g):
    return ndimage.gaussian_filter(g, 1.0)

def denoise_median(g):
    return ndimage.median_filter(g, size=3)

def residual(g, den):
    return g - den(g)

REGIONS = {
    "full": (slice(0,960), slice(0,1280)),
    "photo": (slice(10,750), slice(10,1270)),
    "center": (slice(100,650), slice(100,1180)),
}

def corr(a,b):
    a=a-a.mean(); b=b-b.mean()
    return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()+1e-12))

def test(den_name, den, region_name, norm):
    region = REGIONS[region_name]
    # build references (leave none out first: full ref for diagnostic)
    paths = {p: sorted(glob.glob(f"challenges/task_japan/data/dataset/credited/{p}/*.jpg")) for p in PEOPLE}
    refs = {}
    for p, fs in paths.items():
        acc = np.zeros((960,1280))
        for f in fs:
            g = load_l(f)
            r = residual(g, den)
            if norm == "mul":
                r = r / (den(g) + 1e-4)
            acc += r
        refs[p] = acc / len(fs)
        # zero-mean reference
        refs[p] -= refs[p].mean()

    # correlate each credited photo against all refs (in region)
    correct = 0; total = 0
    margins = []
    for p in PEOPLE:
        for f in paths[p]:
            g = load_l(f)
            r = residual(g, den)
            if norm == "mul":
                r = r / (den(g) + 1e-4)
            rr = r[region] - r[region].mean()
            scores = {}
            for q in PEOPLE:
                rq = refs[q][region]
                scores[q] = corr(rr, rq)
            best = max(scores, key=scores.get)
            total += 1
            if best == p: correct += 1
            sv = sorted(scores.values(), reverse=True)
            margins.append(sv[0]-sv[1])
    print(f"[{den_name}/{region_name}/{norm}] acc={correct}/{total} ({correct/total:.0%}) margin_mean={np.mean(margins):.4f} margin_min={np.min(margins):.4f}")

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv)>1 else "all"
    dens = {"wav": denoise_wav, "wiener": denoise_wiener, "gauss": denoise_gauss, "median": denoise_median}
    for dn,dfn in dens.items():
        for rn in ["photo","full","center"]:
            for norm in ["add","mul"]:
                test(dn, dfn, rn, norm)