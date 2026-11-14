#!/usr/bin/env python3
"""EM-style semi-supervised clustering: iterative reference refinement over all 253 photos."""
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

def load_all():
    items=[]  # (name, residual, true_label or -1)
    for p in PEOPLE:
        for f in sorted(glob.glob(f"challenges/task_japan/data/dataset/credited/{p}/*.jpg")):
            key=f"{p}__{os.path.basename(f)}"
            r=np.load(CACHE/(key+"_g.npy")).astype(np.float32)
            items.append((f"{p}/{os.path.basename(f)}", r-r.mean(), Pidx[p]))
    for f in sorted(glob.glob("challenges/task_japan/data/dataset/anonymous/*.jpg")):
        key=os.path.splitext(os.path.basename(f))[0]
        r=np.load(CACHE/(key+"_g.npy")).astype(np.float32)
        items.append((f"anon/{os.path.basename(f)}", r-r.mean(), -1))
    return items

def main():
    items = load_all()
    N = len(items)
    ncred = sum(1 for _,_,l in items if l>=0)
    R = np.stack([r for _,r,_ in items])  # (N,H,W)
    H,W = R.shape[1], R.shape[2]
    lab = np.array([l for _,_,l in items])
    print(f"N={N} credited={ncred} shape={R.shape}")

    # init references from credited
    refs = np.zeros((6,H,W), np.float32)
    for p in range(6):
        refs[p] = R[np.where(lab==p)[0]].mean(axis=0)

    # iterate
    for it in range(15):
        # assign all to nearest reference (by correlation)
        assign = np.zeros(N, np.int32)
        # compute correlations in blocks to save memory
        S = np.zeros((N,6), np.float32)
        for i in range(N):
            ri = R[i]
            for p in range(6):
                S[i,p] = corr(ri, refs[p])
        assign = S.argmax(axis=1)
        # recompute references from FULL assignment (weighted: credited fixed to truth)
        # keep credited labels fixed to ground truth (supervised anchors)
        assign[:ncred] = lab[:ncred]
        for p in range(6):
            idx = np.where(assign==p)[0]
            refs[p] = R[idx].mean(axis=0)
        # evaluate on credited
        correct = (assign[:ncred]==lab[:ncred]).sum() if it==0 else None
        # report credited purity (did credited stay correct under re-assignment?) only at end
        if it in (0,3,7,14):
            # LOOCV-like: re-assign credited using refs (excluding self roughly)
            c2 = 0
            for i in range(ncred):
                s = np.array([corr(R[i], refs[p]) for p in range(6)])
                c2 += (s.argmax()==lab[i])
            print(f"iter {it}: credited self-consistency {c2}/{ncred}={c2/ncred:.1%}")
    # final
    S = np.zeros((N,6), np.float32)
    for i in range(N):
        for p in range(6):
            S[i,p]=corr(R[i], refs[p])
    assign = S.argmax(axis=1)
    assign[:ncred]=lab[:ncred]
    print("\nFinal cluster sizes:", [int((assign==p).sum()) for p in range(6)])
    # credited self-consistency final
    c2=0; conf=np.zeros((6,6),int)
    for i in range(ncred):
        s=np.array([corr(R[i],refs[p]) for p in range(6)])
        b=s.argmax(); conf[lab[i],b]+=1
        c2+=(b==lab[i])
    print(f"Final credited self-consistency: {c2}/{ncred}={c2/ncred:.1%}")
    for p in range(6): print(f"   {PEOPLE[p]:12s}", list(conf[p]))
    # save assignments
    names=[n for n,_,_ in items]
    np.save("/tmp/opencode/assign.npy", assign)
    with open("/tmp/opencode/names.txt","w") as f:
        f.write("\n".join(names))
    # toorisugari anonymous list
    ts = [(names[i], S[i,0]) for i in range(ncred,N) if assign[i]==0]
    ts.sort(key=lambda x:-x[1])
    print(f"\nToorisugari anonymous ({len(ts)}):")
    for n,s in ts: print(f"   {n} conf={s:.4f}")

if __name__=="__main__":
    main()