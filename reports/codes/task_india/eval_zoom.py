#!/usr/bin/env python3
"""Evaluate multi-zoom pHash candidates on train labels."""
import numpy as np, csv, time

base = '/home/kozak/mllm/MML/work/solutions/task_india/'
D = np.load(base + 'mphash.npz')
names = D['names'].astype(str)
cmp_np = D['mph']; dims = D['dims']
idx = {n: i for i, n in enumerate(names)}
n = len(names)
L = cmp_np.shape[1]
print('n', n, 'levels', L)


def popcount(x):
    x = x.astype(np.uint64)
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int16)


t0 = time.time()
MIN = np.full((n, n), 64, dtype=np.int16)
for i in range(L):
    A = cmp_np[:, i][:, None]
    for j in range(L):
        Dm = popcount(A ^ cmp_np[:, j][None, :])
        np.minimum(MIN, Dm, out=MIN)
print('min-zoom hamming done', time.time() - t0, flush=True)

ar = dims[:, 0] / dims[:, 1]
AR = np.minimum(ar[:, None], ar[None, :]) / np.maximum(ar[:, None], ar[None, :])

rows = list(csv.DictReader(open(base + 'train-output.csv')))
pairs = [(idx[r['image_a']], idx[r['image_b']]) for r in rows
         if r['image_a'] in idx and r['image_b'] in idx]
pa = np.array(pairs)
tr = MIN[pa[:, 0], pa[:, 1]]
trar = AR[pa[:, 0], pa[:, 1]]
print('true pairs', len(pa))
print('multizoom true dist: p50=%.1f p90=%.1f p95=%.1f p99=%.1f max=%.1f' % (
    np.median(tr), np.percentile(tr, 90), np.percentile(tr, 95), np.percentile(tr, 99), tr.max()))
for T in [4, 6, 8, 10, 12, 14, 16, 18, 20]:
    print(f'  T={T}: recall={np.mean(tr<=T):.3f} true={(tr<=T).sum()}')

iu = np.triu_indices(n, 1)
mv = MIN[iu]; av = AR[iu]
print('\nall-pairs multizoom min: p0.01=%.1f p0.1=%.1f p1=%.1f' % (
    np.percentile(mv, 0.01), np.percentile(mv, 0.1), np.percentile(mv, 1)))
for T in [4, 6, 8, 10, 12, 14]:
    for arT in [0.0, 0.95, 0.98, 0.985]:
        cnt = int(((mv <= T) & (av >= arT)).sum())
        print(f'  T={T} ar>={arT}: candidates={cnt}')

np.save(base + 'minzoom.npy', MIN)
