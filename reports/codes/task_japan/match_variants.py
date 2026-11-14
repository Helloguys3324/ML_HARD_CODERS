#!/usr/bin/env python3
"""Matching variants on cached residuals: raw / mle / sign / per-channel-ish. Fast."""
from __future__ import annotations
import glob, os
from pathlib import Path
import numpy as np

PEOPLE = ["toorisugari", "yakin", "amayadori", "shuuden", "jihanki", "rojiura"]
Pidx = {p: i for i, p in enumerate(PEOPLE)}
CACHE = Path("/tmp/opencode/japan_cache")

def corr(a, b):
    a=(a-a.mean()).ravel(); b=(b-b.mean()).ravel()
    return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()+1e-12))

def load_files():
    credited = {p: [] for p in PEOPLE}
    anon = []
    for p in PEOPLE:
        for f in sorted(glob.glob(f"challenges/task_japan/data/dataset/credited/{p}/*.jpg")):
            key=f"{p}__{os.path.basename(f)}"
            r=np.load(CACHE/(key+"_g.npy")).astype(np.float32)
            d=np.load(CACHE/(key+"_gd.npy")).astype(np.float32)
            credited[p].append((os.path.basename(f), r, d))
    for f in sorted(glob.glob("challenges/task_japan/data/dataset/anonymous/*.jpg")):
        key=os.path.splitext(os.path.basename(f))[0]
        r=np.load(CACHE/(key+"_g.npy")).astype(np.float32)
        d=np.load(CACHE/(key+"_gd.npy")).astype(np.float32)
        anon.append((os.path.basename(f), r, d))
    return credited, anon

def feature(r, d, mode):
    if mode == "raw": x = r
    elif mode == "mle": x = r/(d+1e-6)
    elif mode == "sqrt": x = r/np.sqrt(d+1e-6)
    elif mode == "sign": x = np.sign(r)
    else: raise ValueError
    x = x - x.mean()
    return x

def run(mode):
    credited, anon = load_files()
    refs = {}
    for p in PEOPLE:
        xs = [feature(r,d,mode) for _,r,d in credited[p]]
        refs[p] = np.mean(np.stack(xs), axis=0)
    # LOOCV
    correct=0; total=0; conf=np.zeros((6,6),int); margins=[]
    for p in PEOPLE:
        n=len(credited[p]); full=refs[p].copy()
        for i,(name,r,d) in enumerate(credited[p]):
            xi=feature(r,d,mode)
            ref_loo=(full*n - xi)/(n-1)
            scores={q:corr(xi, ref_loo if q==p else refs[q]) for q in PEOPLE}
            best=max(scores,key=scores.get)
            total+=1
            if best==p: correct+=1
            conf[Pidx[p],Pidx[best]]+=1
            sv=sorted(scores.values(),reverse=True); margins.append(sv[0]-sv[1])
    print(f"--- mode={mode}: LOOCV {correct}/{total}={correct/total:.1%} margin={np.mean(margins):.4f}")
    for p in PEOPLE: print(f"    {p:12s}", list(conf[Pidx[p]]))
    # attribution
    from collections import Counter
    cnt=Counter(); detail={}
    for name,r,d in anon:
        xi=feature(r,d,mode)
        scores={p:corr(xi,refs[p]) for p in PEOPLE}
        best=max(scores,key=scores.get)
        cnt[best]+=1; detail[name]=best
    print("    anonymous:", dict(cnt))
    return refs, detail

if __name__=="__main__":
    import sys
    for m in ["raw","mle","sign"]:
        run(m)