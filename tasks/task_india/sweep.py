#!/usr/bin/env python3
"""Offline threshold sweep over the verification records dumped by main.py."""
import csv
import pickle
import itertools
import numpy as np

base = '/home/kozak/mllm/MML/work/solutions/task_india/'
D = pickle.load(open(base + 'verify_full.pkl', 'rb'))
records = D['records']
names = D['names']
dims = D['dims']
truth = set()
for r in csv.DictReader(open(base + 'train-output.csv')):
    a, b = r['image_a'], r['image_b']
    truth.add((a, b) if a <= b else (b, a))
truth_idx = set()
idx = {n: i for i, n in enumerate(names)}
for a, b in truth:
    if a in idx and b in idx:
        i, j = idx[a], idx[b]
        truth_idx.add((min(i, j), max(i, j)))

label = np.array([1 if (min(a, b), max(a, b)) in truth_idx else 0
                  for a, b, *_ in records])
hd = np.array([r[2] for r in records])
inl = np.array([r[3] for r in records])
frac = np.array([r[4] for r in records])
s = np.array([r[5] for r in records])
ncc = np.array([r[6] for r in records])
ar = dims[:, 0] / np.maximum(dims[:, 1], 1)
ar_ratio = np.minimum(ar[[r[0] for r in records]], ar[[r[1] for r in records]]) / \
           np.maximum(ar[[r[0] for r in records]], ar[[r[1] for r in records]])
n_true = len(truth_idx)
print('records', len(records), 'positives', int(label.sum()), 'labelled pairs', n_true)

scale_ok_base = (s >= 0.35) & (s <= 1.45)
degen = (s <= 0.05) & (inl >= 0)

best = []
for min_inl, min_ncc, min_frac, use_strong, strong_inl, strong_hd in itertools.product(
        [12, 16, 20, 26, 30, 35, 40, 50],
        [0.0, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.86],
        [0.3, 0.4, 0.5, 0.6],
        [False, True],
        [0, 10, 20, 30],
        [6, 8, 10]):
    if use_strong and strong_inl not in (0, 10, 20, 30):
        continue
    scale_ok = scale_ok_base | ((s <= 0.05) & (inl >= min_inl))
    geom = (inl >= min_inl) & (frac >= min_frac) & scale_ok & (ncc >= min_ncc)
    if use_strong:
        strong = (hd <= strong_hd) & (ar_ratio >= 0.985) & (inl >= strong_inl)
        pred = geom | strong
    else:
        pred = geom
    tp = int((pred & (label == 1)).sum())
    fp = int((pred & (label == 0)).sum())
    fn = n_true - tp
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    best.append((f1, prec, rec, tp, fp, fn, min_inl, min_ncc, min_frac,
                 use_strong, strong_inl, strong_hd))

best.sort(reverse=True)
print(f'{"F1":>7} {"prec":>7} {"rec":>7} {"TP":>5} {"FP":>5} {"FN":>5}  rule')
for row in best[:25]:
    f1, prec, rec, tp, fp, fn, mi, mn, mf, us, si, sh = row
    print(f'{f1:7.4f} {prec:7.4f} {rec:7.4f} {tp:5d} {fp:5d} {fn:5d}  '
          f'inl>={mi} ncc>={mn} frac>={mf} strong={us} si={si} sh={sh}')

# balanced choice: maximise F1 subject to precision >= 0.95
cand = [r for r in best if r[1] >= 0.95]
print('\nbest with precision >= 0.95:')
for row in cand[:10]:
    f1, prec, rec, tp, fp, fn, mi, mn, mf, us, si, sh = row
    print(f'{f1:7.4f} {prec:7.4f} {rec:7.4f} {tp:5d} {fp:5d} {fn:5d}  '
          f'inl>={mi} ncc>={mn} frac>={mf} strong={us} si={si} sh={sh}')
