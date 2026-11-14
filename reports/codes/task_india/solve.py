#!/usr/bin/env python3
"""Run the India duplicate-pair solver on a labelled directory and report F1.

Usage:
    /home/kozak/mllm/py solve.py [IMAGE_DIR] [LABEL_CSV]

Defaults to the training data shipped with task_india.  Runs main.py in a
scratch directory (so its output.csv does not clobber anything), then compares
the predicted pairs with the ground-truth pairs.
"""
import csv
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = '/home/kozak/mllm/MML/work/tasks/task_india/data/train'
LABELS = '/home/kozak/mllm/MML/work/tasks/task_india/data/train-output.csv'


def norm_pair(a, b):
    return (a, b) if a <= b else (b, a)


def read_pairs(path, cols=('image_a', 'image_b')):
    out = set()
    with open(path, newline='') as fh:
        for r in csv.DictReader(fh):
            a, b = r[cols[0]], r[cols[1]]
            if a and b and a != b:
                out.add(norm_pair(a, b))
    return out


def main(argv):
    img_dir = argv[1] if len(argv) > 1 else DATA
    labels = argv[2] if len(argv) > 2 else LABELS
    work = tempfile.mkdtemp(prefix='twins_eval_')
    cmd = [sys.executable, os.path.join(HERE, 'main.py'), img_dir]
    print('running:', ' '.join(cmd), 'in', work, flush=True)
    subprocess.run(cmd, cwd=work, check=True)
    pred = read_pairs(os.path.join(work, 'output.csv'))
    truth = read_pairs(labels)
    # keep only label pairs whose images are present in the archive
    present = {f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
    truth = {p for p in truth if p[0] in present and p[1] in present}
    tp = len(pred & truth)
    fp = len(pred - truth)
    fn = len(truth - pred)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f'pred={len(pred)} truth={len(truth)} TP={tp} FP={fp} FN={fn}')
    print(f'precision={prec:.4f} recall={rec:.4f} F1={f1:.4f}')
    print('PASS (F1>=0.88)' if f1 >= 0.88 else 'FAIL (F1<0.88)')
    return 0 if f1 >= 0.88 else 2


if __name__ == '__main__':
    sys.exit(main(sys.argv))
