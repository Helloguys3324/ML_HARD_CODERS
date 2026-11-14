#!/usr/bin/env python3
"""Comprehensive PRNU extraction comparison with LOOCV + confusion matrix."""
from __future__ import annotations
import glob, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.restoration import denoise_wavelet
from scipy import ndimage
import cv2

DATA = Path("challenges/task_japan/data/dataset")
PEOPLE = ["toorisugari", "yakin", "amayadori", "shuuden", "jihanki", "rojiura"]
PHOTO = (slice(10, 750), slice(10, 1270))

def load_g(p): return np.asarray(Image.open(p).convert("L"), dtype=np.float64)/255.0
def load_rgb(p): return np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)/255.0

def denoise_wav(g):
    return denoise_wavelet(g, wavelet="db8", method="BayesShrink", mode="soft", wavelet_levels=5, rescale_sigma=True)

def denoise_nlm(g):
    u8 = np.clip(g*255,0,255).astype(np.uint8)
    d = cv2.fastNlMeansDenoising(u8, None, 7, 7, 21)
    return d.astype(np.float64)/255.0

def denoise_wiener(g):
    from scipy.signal import wiener
    return wiener(g, mysize=3)

def corr(a, b):
    a = (a-a.mean()).ravel(); b=(b-b.mean()).ravel()
    return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()+1e-12))

def loocv(method, channels="gray"):
    t0=time.time()
    credited = {p: sorted(glob.glob(f"challenges/task_japan/data/dataset/credited/{p}/*.jpg")) for p in PEOPLE}
    if method=="wav": den=denoise_wav
    elif method=="nlm": den=denoise_nlm
    elif method=="wiener": den=denoise_wiener

    # precompute residual per photo
    R = {}
    for p, fs in credited.items():
        Rs=[]
        for f in fs:
            if channels=="gray":
                g=load_g(f); r=(g-den(g))[PHOTO]; Rs.append(r)
            else:
                rgb=load_rgb(f); rr=[]
                for c in range(3):
                    gc=rgb[:,:,c]; rr.append((gc-den(gc))[PHOTO])
                Rs.append(np.stack(rr,axis=0))  # (3,H,W)
        R[p]=Rs

    refs={}
    for p in PEOPLE:
        if channels=="gray":
            r=np.mean(np.stack(R[p]),axis=0)
        else:
            r=np.mean(np.stack(R[p]),axis=0)  # (3,H,W)
        refs[p]=r

    def ccorr(a,b):
        if channels=="gray":
            return corr(a,b)
        s=0
        for c in range(3):
            s+=corr(a[c],b[c])
        return s/3

    correct=0; total=0; margins=[]; conf=np.zeros((6,6),dtype=int)
    Pidx={p:i for i,p in enumerate(PEOPLE)}
    for p in PEOPLE:
        n=len(R[p]); idx=Pidx[p]
        for i in range(n):
            r=R[p][i]
            # LOO ref
            rr = r - (r.mean(axis=(-2,-1), keepdims=True) if channels!="gray" else r.mean())
            for c in (range(3) if channels!="gray" else [None]):
                pass
            # simpler: subtract this residual's contribution from full ref
            full = refs[p]
            if channels=="gray":
                ref_loo = (full*n - (r-r.mean()))/(n-1)
            else:
                rm = r - r.mean(axis=(-2,-1),keepdims=True)
                ref_loo = (full*n - rm)/(n-1)
            scores={}
            for q in PEOPLE:
                scores[q]=ccorr(r if channels=="gray" else r, ref_loo if q==p else refs[q])
            best=max(scores,key=scores.get)
            total+=1
            if best==p: correct+=1
            conf[idx,Pidx[best]]+=1
            sv=sorted(scores.values(),reverse=True)
            margins.append(sv[0]-sv[1])
    print(f"[{method}/{channels}] LOOCV {correct}/{total} = {correct/total:.1%} margin {np.mean(margins):.4f} ({time.time()-t0:.0f}s)")
    print("  confusion (rows=truth):", Pidx)
    for p in PEOPLE:
        print(f"    {p:12s}", list(conf[Pidx[p]]))
    return correct/total

if __name__=="__main__":
    for m in sys.argv[1:] if len(sys.argv)>1 else ["wav","wiener"]:
        loocv(m, "gray")
    # wav per-channel
    loocv("wav", "rgb")