"""Local streaming evaluation for task_albania anomaly detector.

Simulates the organizer harness: feeds one reading at a time, collects
scores, thresholds at 0.50, computes overall F1 + per-type recall + final score.

The detector under test is imported from detector.py (the actual deliverable).
"""
import math
import numpy as np
import pandas as pd
from collections import deque

from detector import Detector


def evaluate(detector_cls=Detector, data_dir="challenges/task_albania/data"):
    inp = pd.read_csv(f"{data_dir}/train-input.csv")
    out = pd.read_csv(f"{data_dir}/train-output.csv")
    det = detector_cls()
    scores = []
    for rec in inp.to_dict("records"):
        scores.append(det.update(rec))
    scores = np.asarray(scores, dtype=float)
    labels = out["is_anomaly"].values.astype(int)
    types = out["anomaly_type"].values

    # threshold
    pred = (scores > 0.5).astype(int)

    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    def recall(t):
        sub = (types == t)
        if sub.sum() == 0:
            return 1.0
        return ((pred[sub] == 1).sum()) / sub.sum()

    r_point = recall("point")
    r_drift = recall("drift")
    r_coll = recall("collective")
    score = 100 * (0.80 * f1 + 0.05 * r_point + 0.075 * r_drift + 0.075 * r_coll)
    score = min(100.0, max(1.0, score))

    print(f"tp={tp} fp={fp} fn={fn} tn={tn}")
    print(f"precision={prec:.4f} recall={rec:.4f} F1={f1:.4f}")
    print(f"recall point={r_point:.4f} drift={r_drift:.4f} collective={r_coll:.4f}")
    print(f"SCORE = {score:.2f}")
    return score, scores, labels, types


if __name__ == "__main__":
    evaluate()