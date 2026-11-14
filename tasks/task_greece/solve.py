"""task_greece "Just a siblings' thing" — SOLVED (do not submit from this script).

Three PyTorch linear oracles are evaluated on the supplied test set at the fixed
decision threshold 0.50. All three achieve exactly the same accuracy (0.95), so
the business statement is the tie-breaker:

    "missing a true catastrophic threat is far more expensive than raising a
     false alarm"

That is a cost-sensitive setting with C_FN >> C_FP. The appropriate metric is
therefore RECALL/SENSITIVITY (equivalently: minimise false negatives, maximise
F-beta with beta>1, balanced accuracy, MCC, PR-AUC, ROC-AUC). The oracle with the
fewest missed catastrophic threats wins.

Result:
    ARES   : acc .95, recall .95, FN  5, FP 45
    ATHENA : acc .95, recall .75, FN 25, FP 25
    APOLLO : acc .95, recall .55, FN 45, FP  5
    -> ARES dominates every recall-oriented metric. Flag: SIGMOID_ARES

Run with: /home/kozak/mllm/py solve.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

DATA = Path("/home/kozak/mllm/MML/work/tasks/task_greece/data")
OUT = Path("/home/kozak/mllm/MML/work/solutions/task_greece")
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(DATA))
from model import FEATURE_COLUMNS, MythicThreatClassifier  # noqa: E402

MODELS = ["ares", "athena", "apollo"]


def load_model(name: str) -> MythicThreatClassifier:
    model = MythicThreatClassifier()
    state = torch.load(DATA / "models" / f"model_{name}.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state)  # strict=True -> keys must match model.py
    model.eval()
    return model


def main() -> None:
    df = pd.read_csv(DATA / "test_dataset.csv")
    labels = pd.read_csv(DATA / "test_labels.csv")
    assert (df["encounter_id"].to_numpy() == labels["encounter_id"].to_numpy()).all()
    y = labels["is_catastrophic_threat"].to_numpy().astype(int)
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    rows, probs = [], {}
    print("=" * 78)
    print("Checkpoint contents (static, weights_only=True):")
    for name in MODELS:
        model = load_model(name)
        w = model.linear.weight.detach().numpy().ravel()
        b = float(model.linear.bias.detach().numpy().ravel()[0])
        nz = np.flatnonzero(w)
        print(f"  {name:7s} w = {np.array2string(w, precision=2)}  bias = {b:.1f}")
        print(f"          -> uses feature(s) {[FEATURE_COLUMNS[i] for i in nz]} "
              f"(w={w[nz[0]]:.1f}); decision boundary is that feature >= 0")

        logits = model(torch.from_numpy(X)).detach().numpy()
        p = 1.0 / (1.0 + np.exp(-logits))  # sigmoid, threshold p >= 0.50
        probs[name] = p
        pred = (p >= 0.50).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append({
            "model": name.upper(),
            "feature": FEATURE_COLUMNS[int(nz[0])],
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall_sensitivity": recall_score(y, pred),
            "specificity": tn / (tn + fp),
            "f1": f1_score(y, pred),
            "f2": fbeta_score(y, pred, beta=2),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "mcc": matthews_corrcoef(y, pred),
            "roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "n_pred_pos": int(pred.sum()), "mean_prob": float(p.mean()),
        })

    table = pd.DataFrame(rows).set_index("model")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 60)
    print("=" * 78)
    print("Metrics at threshold 0.50 (dataset: 1000 rows, 100 positive / 900 negative):")
    print(table.round(4).T.to_string())
    table.to_csv(OUT / "metrics.csv")

    print("=" * 78)
    print("Cost-sensitive analysis: cost = FP + k*FN  (k = C_FN / C_FP)")
    for k in [0.5, 1.0, 2.0, 5.0, 10.0]:
        costs = {m.upper(): table.loc[m.upper(), "fp"] + k * table.loc[m.upper(), "fn"] for m in MODELS}
        best = min(costs, key=costs.get)
        print(f"  k={k:>4}: " + "  ".join(f"{m}={c:6.1f}" for m, c in costs.items()) + f"  -> {best}")
    print("  For every k>1 (missed threat costs more than false alarm) ARES is optimal; "
          "at k=1 all three tie (hence identical accuracy).")

    # Joint prediction structure: nested positive sets (precision/recall trade-off).
    pred_df = pd.DataFrame({m: (probs[m] >= 0.50).astype(int) for m in MODELS})
    pred_df["y"] = y
    print("=" * 78)
    print("Joint prediction counts (ares, athena, apollo, y) -> n:")
    print(pred_df.groupby(list(pred_df.columns)).size().to_string())
    print(f"Positive sets nested: apollo⊂athena⊂ares? "
          f"{bool(((pred_df.apollo <= pred_df.athena).all()) and ((pred_df.athena <= pred_df.ares).all()))}")

    print("=" * 78)
    print("Decision rule from story: C_FN >> C_FP  =>  maximise RECALL / minimise FN")
    best = table["recall_sensitivity"].idxmax()
    print(f"Highest recall = {best}  (FN={int(table.loc[best, 'fn'])})")
    assert table["accuracy"].nunique() == 1, "story premise: identical accuracy"
    print(f"All accuracies identical: {table['accuracy'].unique().tolist()}")
    print(f"\nANSWER: SIGMOID_{best}")

    json.dump(
        {
            "metrics_at_0.50": table.round(6).to_dict(orient="index"),
            "selected_model": best,
            "answer": f"SIGMOID_{best}",
            "reason": "C_FN >> C_FP => maximise recall / minimise false negatives",
        },
        open(OUT / "metrics.json", "w"),
        indent=2,
    )


if __name__ == "__main__":
    main()
