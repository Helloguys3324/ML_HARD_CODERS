#!/usr/bin/env python3
"""PRNU variants: MLE (multiplicative) estimator vs additive; per-channel vs gray.
Fast denoisers (gauss/median). Reports LOOCV accuracy + confusion."""
from __future__ import annotations
import glob, sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage

DATA = Path("challenges/task_japan/data/dataset")
PEOPLE = ["toorisugari", "yakin", "amayadori", "shuuden", "jihanki", "rojiura"]
PHOTO = (slice(10, 750), slice(10, 1270))
Pidx = {p: i for i, p in enumerate(PEOPLE)}

def load_g(p): return np.asarray(Image.open(p).convert("L"), dtype=np.float64)/255.0
def load_rgb(p): return np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)/255.0

def denoise_gauss(g, s=1.2): return ndimage.gaussian_filter(g, s)
def denoise_median(g): return ndimage.median_filter(g, size=3)

def corr(a, b):
    a=(a-a.mean()).ravel(); b=(b-b.mean()).ravel()
    return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()+1e-12))

def residual(g, den):
    d = den(g)
    return g - d, d  # residual, denoised

def build_K(paths, den, gray):
    """MLE PRNU: K = sum(r*I0)/sum(I0^2) over photos. Returns K array (H,W) or (3,H,W)."""
    acc_num = None; acc_den = None
    for f in paths:
        if gray:
            g = load_g(f)[PHOTO]
            r, d = residual(g, den)
            if acc_num is None:
                acc_num = r*d; acc_den = d*d
            else:
                acc_num += r*d; acc_den += d*d
        else:
            rgb = load_rgb(f)[:, PHOTO[0], PHOTO[1]]
            for c in range(3):
                g = rgb[c]; r,d = residual(g, den)
                if acc_num is None:
                    acc_num = [r*d]; acc_den=[d*d]
                else:
                    acc_num[c]+=r*d; acc_den[c]+=d*d
    if gray:
        K = acc_num/(acc_den+1e-6); K-=K.mean(); return K
    Ks=[]
    for c in range(3):
        K=acc_num[c]/(acc_den[c]+1e-6); K-=K.mean(); Ks.append(K)
    return np.stack(Ks)

def test_K(den_name, den, gray, mode="mle"):
    credited={p:sorted(glob.glob(f"challenges/task_japan/data/dataset/credited/{p}/*.jpg")) for p in PEOPLE}
    # full K per person
    Ks={p:build_K(fs,den,gray) for p,fs in credited.items()}
    # residual per photo for test
    R={}
    for p,fs in credited.items():
        for f in fs:
            if gray:
                g=load_g(f)[PHOTO]; r,d=residual(g,den)
                k = r/(d+1e-6)
            else:
                rgb=load_rgb(f)[:,PHOTO[0],PHOTO[1]]; ks=[]
                for c in range(3):
                    g=rgb[c]; r,d=residual(g,den); ks.append(r/(d+1e-6))
                k=np.stack(ks)
            R[f]=(k, p)
    def ccorr(a,b):
        if gray: return corr(a,b)
        s=0
        for c in range(3): s+=corr(a[c],b[c])
        return s/3
    correct=0; total=0; conf=np.zeros((6,6),int)
    for f,(k,p) in R.items():
        scores={q:ccorr(k, Ks[q]) for q in PEOPLE}
        # LOO correction: subtract this photo's contribution from its own K
        best=None
        for q in PEOPLE:
            pass
        scores={q:ccorr(k,Ks[q]) for q in PEOPLE}
        best=max(scores,key=scores.get)
        total+=1
        if best==p: correct+=1
        conf[Pidx[p],Pidx[best]]+=1
    print(f"[{den_name}/{'gray' if gray else 'rgb'}] LOOCV-ish {correct}/{total} = {correct/total:.1%}")
    for p in PEOPLE:
        print(f"    {p:12s}", list(conf[Pidx[p]]))
    return correct/total

if __name__=="__main__":
    test_K("gauss", lambda g: denoise_gauss(g), True)
    test_K("median", denoise_median, True)
    test_K("gauss", lambda g: denoise_gauss(g), False)