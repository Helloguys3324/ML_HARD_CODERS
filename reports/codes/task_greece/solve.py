"""task_greece — «Just a siblings' thing» (Medium, 200 pts).

Three binary classifiers (Ares, Athena, Apollo) share the same architecture
(MythicThreatClassifier, a single linear layer over 9 features) and report the
SAME test accuracy, so accuracy alone cannot pick a winner.

The proposed business case is explicit: "missing a true catastrophic threat is
far more expensive than raising a false alarm" -> FALSE NEGATIVES are the costly
error.  The metric that directly penalises false negatives is RECALL (a.k.a.
sensitivity / true-positive rate).  With class imbalance (which we verify below)
accuracy is misleading, and the case is FN-averse, so we select the model with
the highest recall at the fixed decision threshold 0.50.

We still compute a full metric table (accuracy, precision, recall, F1,
balanced accuracy, MCC, ROC-AUC, PR-AUC, calibration) as evidence.
"""
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, roc_auc_score,
    average_precision_score, confusion_matrix,
)

DATA = "challenges/task_greece/data"
THRESH = 0.50

FEATURE_COLUMNS = [
    "creature_mass_talents", "oracle_omen", "aggression_index", "divine_aura",
    "aegis_resonance", "armor_resistance", "witness_count",
    "styx_disturbance", "omen_intensity",
]


class MythicThreatClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(len(FEATURE_COLUMNS), 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def main():
    df = pd.read_csv(f"{DATA}/test_dataset.csv")
    labels = pd.read_csv(f"{DATA}/test_labels.csv")
    y_true = labels["is_catastrophic_threat"].to_numpy()
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    print("Class distribution: 0 =", int((y_true == 0).sum()),
          " 1 =", int((y_true == 1).sum()),
          " imbalance ratio =", round((y_true == 0).mean() / (y_true == 1).mean(), 3))

    rows = []
    for name in ["ares", "athena", "apollo"]:
        ckpt = torch.load(f"{DATA}/models/model_{name}.pth",
                          map_location="cpu", weights_only=False)
        model = MythicThreatClassifier()
        model.load_state_dict(ckpt)
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(X))
        probs = torch.sigmoid(logits).numpy()
        preds = (probs >= THRESH).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_true, preds),
            "precision": precision_score(y_true, preds, zero_division=0),
            "recall": recall_score(y_true, preds, zero_division=0),
            "f1": f1_score(y_true, preds, zero_division=0),
            "bal_acc": balanced_accuracy_score(y_true, preds),
            "mcc": matthews_corrcoef(y_true, preds),
            "roc_auc": roc_auc_score(y_true, probs),
            "pr_auc": average_precision_score(y_true, probs),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        })

    table = pd.DataFrame(rows)
    print("\nMetric table @ threshold 0.50:")
    print(table.to_string(index=False))

    # FN = the expensive error in this case -> pick the highest recall.
    best = table.sort_values(
        by=["recall", "f1", "pr_auc"], ascending=[False, False, False]
    ).iloc[0]
    print("\nCase: false negatives (missing a real threat) are the costly error.")
    print("=> Relevant metric: RECALL.  Best model:", best["model"].upper())
    print("Flag: SIGMOID_" + best["model"].upper())

    table.to_csv(f"{DATA}/../metrics_table.csv", index=False)
    return best["model"].upper()


if __name__ == "__main__":
    main()