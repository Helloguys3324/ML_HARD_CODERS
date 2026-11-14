#!/usr/bin/env python3
"""Analyze train labels vs hash features (development helper)."""
import numpy as np, csv, sys

D = np.load('/home/kozak/mllm/MML/work/solutions/task_india/feats.npz', allow_pickle=False)
names = D['names'].astype(str)
idx = {n: i for i, n in enumerate(names)}
ph, dh, ah, g32, hist, dims = D['phash'], D['dhash'], D['ahash'], D['g32'], D['hist'], D['dims']

rows = list(csv.DictReader(open('/home/kozak/mllm/MML/work/solutions/task_india/train-output.csv')))
pairs = [(idx[r['image_a']], idx[r['image_b']]) for r in rows
         if r['image_a'] in idx and r['image_b'] in idx]
print('n images', len(names), 'labelled pairs available', len(pairs))


def popcount(x):
    x = x.astype(np.uint64)
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int32)


def hamming_mat(a):
    n = len(a)
    M = np.zeros((n, n), dtype=np.int16)
    step = 512
    for s in range(0, n, step):
        A = a[s:s + step][:, None]
        M[s:s + step] = popcount(A ^ a[None, :])
    return M


print('computing hamming matrices...', flush=True)
HP = hamming_mat(ph)
HD = hamming_mat(dh)
HA = hamming_mat(ah)

G = g32 @ g32.T
Hh = hist @ hist.T
ar = dims[:, 0] / dims[:, 1]
AR = np.minimum(ar[:, None], ar[None, :]) / np.maximum(ar[:, None], ar[None, :])

pa = np.array(pairs)
lab = {'hp': HP[pa[:, 0], pa[:, 1]], 'hd': HD[pa[:, 0], pa[:, 1]],
       'ha': HA[pa[:, 0], pa[:, 1]], 'g': G[pa[:, 0], pa[:, 1]],
       'hist': Hh[pa[:, 0], pa[:, 1]], 'ar': AR[pa[:, 0], pa[:, 1]]}
print('TRUE pairs distributions:')
for k, v in lab.items():
    print(f'  {k}: min={v.min():.3f} p1={np.percentile(v,1):.3f} p5={np.percentile(v,5):.3f} '
          f'median={np.median(v):.3f} p95={np.percentile(v,95):.3f} max={v.max():.3f}')

# all-pairs stats for negatives
iu = np.triu_indices(len(names), 1)
print('\nALL pairs quantiles (negatives dominate):')
for name, M in [('phash', HP), ('dhash', HD), ('ahash', HA), ('g32dot', G), ('hist', Hh), ('ar', AR)]:
    v = M[iu]
    print(f'  {name}: p0.001={np.percentile(v,0.001):.3f} p0.01={np.percentile(v,0.01):.3f} '
          f'p0.1={np.percentile(v,0.1):.3f} p1={np.percentile(v,1):.3f} median={np.median(v):.3f}')

# how many all-pairs exceed true-pair max thresholds
for th in [6, 8, 10, 12, 14, 16, 20]:
    print(f'phash<={th}: all pairs {int((HP[iu] <= th).sum())}')

np.savez('/home/kozak/mllm/MML/work/solutions/task_india/mats.npz', HP=HP, HD=HD, HA=HA,
         G=G.astype(np.float32), H=Hh.astype(np.float32), AR=AR.astype(np.float32))
