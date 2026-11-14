#!/usr/bin/env python3
"""Tune classifier using direct-NCC + SIFT aligned features. Reads warp_features.pkl."""
import pickle, os, numpy as np, csv, glob, re

BASE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(BASE, 'data', 'train')

wf = pickle.load(open(os.path.join(BASE, 'warp_features.pkl'), 'rb'))
print("n records", len(wf))

# columns per pair
feat_names = ['nmatch', 'ninl', 'ratio', 'ncc', 'ncc_cc', 'mae', 'mae_cc', 'scale', 'shear', 'rot']
X = []
y = []
keys = []
for (a, b), v in wf.items():
    row = [v.get(f, np.nan) for f in feat_names]
    X.append(row)
    y.append(v['label'])
    keys.append((a, b, v.get('ncc'), v.get('ratio'), v.get('ncc_cc')))

X = np.array(X, dtype=float)
y = np.array(y)
print("total", len(y), "pos", y.sum())
T = X[y == 1]; F = X[y == 0]
print(f"TRUE {len(T)}  FALSE {len(F)}")
for ci, nm in enumerate(feat_names):
    tv = T[:, ci]; fv = F[:, ci]
    if nm in ('ncc', 'ncc_cc', 'ratio', 'scale', 'shear'):
        print(f"{nm:8s} TRUE  min {np.nanmin(tv):.4f} p5 {np.nanpercentile(tv,5):.4f} med {np.nanmedian(tv):.4f} p95 {np.nanpercentile(tv,95):.4f} max {np.nanmax(tv):.4f}")
        print(f"{'':8s} FALSE min {np.nanmin(fv):.4f} p5 {np.nanpercentile(fv,5):.4f} med {np.nanmedian(fv):.4f} p95 {np.nanpercentile(fv,95):.4f} max {np.nanmax(fv):.4f}")

# simple single-feature threshold eval
def eval_th(v, lab, th, gt=True):
    pred = (v > th) if gt else (v < th)
    tp = ((pred == 1) & (lab == 1)).sum(); fp = ((pred == 1) & (lab == 0)).sum()
    fn = ((pred == 0) & (lab == 1)).sum()
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return f1, prec, rec

print("\n=== single-feature best F1 ===")
for ci, nm in enumerate(feat_names):
    v = X[:, ci].copy()
    ok = ~np.isnan(v)
    best = (0, 0)
    lo, hi = np.nanpercentile(v[ok], 1), np.nanpercentile(v[ok], 99)
    for th in np.linspace(lo, hi, 200):
        f1, p, r = eval_th(v[ok], y[ok], th, gt=True)
        if f1 > best[0]: best = (f1, th, p, r)
        f1, p, r = eval_th(v[ok], y[ok], th, gt=False)
        if f1 > best[0]: best = (f1, th, p, r)
    print(f"{nm:8s} best F1={best[0]:.4f} @ {best[1]:.3f} (prec {best[2]:.3f} rec {best[3]:.3f})")