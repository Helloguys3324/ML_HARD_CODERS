#!/usr/bin/env python3
"""Full 253x253 correlation clustering of photo residuals, evaluated vs credited labels."""
from __future__ import annotations
import glob, os
from pathlib import Path
import numpy as np

DATA = Path("challenges/task_japan/data/dataset")
PEOPLE = ["toorisugari", "yakin", "amayadori", "shuuden", "jihanki", "rojiura"]
CACHE = Path("/tmp/opencode/japan_cache")
Pidx = {p: i for i, p in enumerate(PEOPLE)}

def load_all():
    rows = []; labels = []; names = []
    for p in PEOPLE:
        for f in sorted(glob.glob(f"challenges/task_japan/data/dataset/credited/{p}/*.jpg")):
            key = f"{p}__{os.path.basename(f)}"
            r = np.load(CACHE / (key + "_g.npy")).astype(np.float32)
            rows.append(r); labels.append(Pidx[p]); names.append(f"{p}/{os.path.basename(f)}")
    for f in sorted(glob.glob("challenges/task_japan/data/dataset/anonymous/*.jpg")):
        key = os.path.splitext(os.path.basename(f))[0]
        r = np.load(CACHE / (key + "_g.npy")).astype(np.float32)
        rows.append(r); labels.append(-1); names.append(f"anon/{os.path.basename(f)}")
    R = np.stack(rows)  # (N, H, W)
    N, H, W = R.shape
    R = R.reshape(N, -1)
    # center each row
    R = R - R.mean(axis=1, keepdims=True)
    # normalize each row to unit norm (cosine)
    nrm = np.linalg.norm(R, axis=1, keepdims=True)
    nrm[nrm == 0] = 1
    R = R / nrm
    return R.astype(np.float32), np.array(labels), names

def main():
    R, labels, names = load_all()
    N = R.shape[0]
    print(f"Loaded {N} residuals, dim {R.shape[1]}", flush=True)
    # cosine similarity gram matrix
    S = R @ R.T  # (N,N) float32
    S = S.astype(np.float64)
    np.fill_diagonal(S, 0.0)
    # known labels for credited (first 108)
    cred_mask = labels >= 0
    ncred = cred_mask.sum()
    # evaluate: for each credited photo, nearest-credited-neighbor (leave-one-out) same-label fraction
    # Build similarity among credited only
    Sc = S[:ncred, :ncred]
    # LOOCV: assign each credited to the person with highest mean similarity (excluding self)
    correct = 0; total = 0; conf = np.zeros((6,6), int)
    # reference = mean of normalized residuals per person among credited (excluding self)
    for i in range(ncred):
        tru = labels[i]
        sims = np.zeros(6)
        for p in range(6):
            idx = np.where((labels[:ncred] == p) & (np.arange(ncred) != i))[0]
            sims[p] = Sc[i, idx].mean()
        best = sims.argmax()
        total += 1
        if best == tru: correct += 1
        conf[tru, best] += 1
    print(f"Credited LOOCV (via mean cosine to references): {correct}/{total} = {correct/total:.1%}")
    for p in range(6):
        print(f"   {PEOPLE[p]:12s}", list(conf[p]))

    # clustering: spectral via k-means on similarity
    from sklearn.cluster import KMeans
    # use S (similarity) as features for spectral embedding
    # simple: kmeans on the normalized residual space? too high dim. Use spectral.
    D = 1.0 - S  # dissimilarity
    # spectral embedding via SVD of centered similarity
    from scipy.sparse.csgraph import laplacian
    from scipy import linalg
    # symmetric normalized laplacian
    # Use np for spectral: compute top eigenvectors of normalized similarity
    d = S.sum(axis=1)
    d[d==0]=1
    inv = 1.0/np.sqrt(d)
    L = np.eye(N) - (inv[:,None]*S*inv[None,:])
    evals, evecs = linalg.eigh(L)
    embed = evecs[:, 1:7]  # skip first (0 eigenvalue)
    ks = KMeans(n_clusters=6, n_init=10, random_state=0).fit(embed)
    cl = ks.labels_
    # map clusters to people by majority credited label
    for c in range(6):
        idx = np.where((cl[:ncred]==c))[0]
        if len(idx)==0: continue
        cnt = np.bincount(labels[idx], minlength=6)
        top = PEOPLE[cnt.argmax()]
        n_anon = (cl[ncred:]==c).sum()
        print(f"cluster {c}: n_cred={len(idx)} top={top} counts={cnt.tolist()} n_anon={n_anon}")
    # save
    np.save("/tmp/opencode/S.npy", S)
    np.save("/tmp/opencode/clusters.npy", cl)
    np.save("/tmp/opencode/labels.npy", labels)
    with open("/tmp/opencode/names.txt","w") as f:
        f.write("\n".join(names))
    print("saved")

if __name__ == "__main__":
    main()