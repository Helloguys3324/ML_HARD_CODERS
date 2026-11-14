#!/usr/bin/env python3
"""PRNU attribution test — compare multiple denoisers and metrics."""
from __future__ import annotations
import sys, time, glob
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.restoration import denoise_wavelet

DATA = Path("challenges/task_japan/data/dataset")
PEOPLE = ["toorisugari", "yakin", "amayadori", "shuuden", "jihanki", "rojiura"]

def load_gray(p):
    return np.asarray(Image.open(p).convert("L"), dtype=np.float64)/255.0

def load_rgb(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)/255.0

def wavelet_residual(g):
    d = denoise_wavelet(g, method="BayesShrink", mode="soft", wavelet_levels=4, rescale_sigma=True)
    return g - d

def gauss_residual(g):
    return g - ndimage.gaussian_filter(g, 1.5)

def whiten(x):
    x = x - x.mean()
    return x

def corrcoef(a,b):
    a=a.ravel(); b=b.ravel()
    a=a-a.mean(); b=b-b.mean()
    return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()+1e-12))

def run(dtype, residual_fn, name):
    credited = {}
    for p in PEOPLE:
        fs = sorted(glob.glob(f"challenges/task_japan/data/dataset/credited/{p}/*.jpg"))
        credited[p]=fs
    # references
    refs={}
    for p,fs in credited.items():
        acc=None
        for f in fs:
            if dtype=="gray":
                g=load_gray(f); r=whiten(residual_fn(g))
            else:
                r=[]
                for c in range(3):
                    ch=np.asarray(Image.open(f).convert("RGB")[:,:,c],dtype=np.float64)/255.0 if False else None
            if acc is None: acc=r
            else: acc=acc+r
        refs[p]=acc/len(fs)
    # attribute
    anon=sorted(glob.glob("challenges/task_japan/data/dataset/anonymous/*.jpg"))
    from collections import Counter
    cnt=Counter(); correct={}
    detail=[]
    for f in anon:
        if dtype=="gray":
            r = whiten(residual_fn(load_gray(f)))
        scores={p:corrcoef(r,refs[p]) for p in PEOPLE}
        best=max(scores,key=scores.get)
        cnt[best]+=1
        detail.append((Path(f).name,best,scores))
    print(f"=== [{name}] counts:", dict(cnt))
    # margin analysis
    return refs, detail

if __name__=="__main__":
    for name,fn in [("wavelet",wavelet_residual),("gauss",gauss_residual)]:
        t=time.time()
        run("gray",fn,name)
        print(f"  time {time.time()-t:.1f}s")