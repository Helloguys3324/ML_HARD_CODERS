"""
Audit and Solution Write-Up for task_estonia (This Data Is Perfect!)
=====================================================================

1. The Real Bug and the Two Decoys
-----------------------------------
In the reported pipeline (`project_pipeline.py`), the fraud-detection team claimed
a 93.42% accuracy using a 1-Nearest-Neighbor classifier on 6 standardized numerical features.
The audit revealed three choices that look questionable at first glance:

- Decoy 1: Pre-split Feature Scaling (StandardScaler fitted on entire dataset)
  `X_scaled = StandardScaler().fit_transform(X)` before `train_test_split(...)`.
  Practitioners immediately flag this as textbook data leakage (test set distribution
  statistics leaking into the scaler).
  Evidence: Fitting StandardScaler strictly on the training fold yields 93.67% accuracy.
  The score remains virtually unchanged (+0.25%, a 1-sample difference on 395 test rows).
  This proves pre-split scaling is NOT what inflated the score.

- Decoy 2: Evaluation on a Single Random Train/Test Split (Evaluation Variance / Seed)
  `train_test_split(..., test_size=0.2, random_state=1137)` without cross-validation or stratification.
  One might hypothesize that random_state=1137 was cherry-picked or an artifact of an 80/20 split.
  Evidence: Running full row-level Leave-One-Out CV (`LeaveOneOut()`) across all 1971 rows
  yields 95.94% accuracy (and stratified split yields 93.92%). The score is still heavily
  inflated (~94-96%), proving the single split was NOT the root cause.

- The Real Bug: Identity Dropping Without Grouped Splitting (Group / Duplicate Identity Leakage)
  The pipeline dropped `date_of_birth` and `citizenship`:
  `X = df.drop(columns=["submission_id", "date_of_birth", "citizenship", "label"])`
  under the assumption that personal identity fields should not be used in modeling.
  However, the dataset contains multiple applications submitted by the SAME applicant
  (identified by identical `date_of_birth` and `citizenship`). Out of 1971 applications,
  there are only 900 unique applicants (1611 applications are repeats, 2 to 4 per person).
  Crucially, features per applicant are virtually identical (same distance, timing, scanner,
  within slight jitter: mean distance between same-applicant rows is 0.038 vs 3.365 across
  different applicants), and the fraud label is consistent per person.
  Because the random split ignores applicant identity, 78.0% of the test set rows have another
  submission from the same applicant in the training set.
  A 1-Nearest-Neighbor classifier looks up the nearest point, matches the identical applicant's
  earlier submission with ~0 distance, and achieves 100.0% accuracy on those leaked rows!

2. The Corrected Pipeline
--------------------------
To evaluate the true performance of the fraud screen on unseen applicants:
- Maintain the shipped model: `StandardScaler() + KNeighborsClassifier(n_neighbors=1)`.
- Use `LeaveOneGroupOut` where groups are defined by applicant identity (`date_of_birth` + `citizenship`).
- In each fold, all applications from a person are held out together. The scaler and 1-NN model
  are fitted on other applicants, preventing any applicant lookup leakage.
- Generate out-of-fold predictions using `cross_val_predict(..., cv=LeaveOneGroupOut())`.

3. True Performance Figure
---------------------------
- Correct predictions: 1511 / 1971 rows.
- True Accuracy: 0.766616 (76.66%).
- As a whole percent: 77% -> Flag: `SIGMOID_77`.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    train_test_split,
    LeaveOneOut,
    LeaveOneGroupOut,
    cross_val_predict,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_audit(data_path: Path):
    print("=" * 70)
    print("LMML 2026: task_estonia (This Data Is Perfect!) - ML Leakage Audit")
    print("=" * 70)

    df = pd.read_csv(data_path)
    print(f"\n[+] Loaded dataset: {len(df)} rows, {len(df.columns)} columns.")

    # Features and identity groups
    feat_cols = [
        "form_fill_seconds",
        "doc_scan_quality",
        "photo_match_score",
        "email_domain_age_days",
        "ip_geo_distance_km",
        "address_history_years",
    ]
    X = df[feat_cols]
    y = df["label"]
    groups = df["date_of_birth"].astype(str) + "_" + df["citizenship"].astype(str)

    n_unique_applicants = groups.nunique()
    print(f"[+] Total applications: {len(df)}")
    print(f"[+] Unique applicant identities (DOB + Citizenship): {n_unique_applicants}")

    # Baseline: Shipped pipeline
    X_scaled = StandardScaler().fit_transform(X)
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X_scaled, y, df.index, test_size=0.2, random_state=1137
    )
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_tr, y_tr)
    shipped_acc = accuracy_score(y_te, clf.predict(X_te))
    print(f"\n1. Shipped pipeline (pre-split scaling, 80/20 split):")
    print(f"   Reported test accuracy: {shipped_acc:.4f} ({shipped_acc*100:.2f}%)")

    # Decoy 1: Fix pre-split scaling in isolation
    X_tr_raw, X_te_raw, y_tr_raw, y_te_raw = train_test_split(
        X, y, test_size=0.2, random_state=1137
    )
    pipe = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=1))])
    pipe.fit(X_tr_raw, y_tr_raw)
    decoy1_acc = accuracy_score(y_te_raw, pipe.predict(X_te_raw))
    print(f"\n2. Decoy 1: Fixing pre-split scaling in isolation (fit on train only):")
    print(f"   Accuracy: {decoy1_acc:.4f} ({decoy1_acc*100:.2f}%)")
    print(f"   Delta vs shipped: {decoy1_acc - shipped_acc:+.4f} (virtually identical, score is NOT fake due to scaling)")

    # Decoy 2: Fixing single split via Leave-One-Out CV (ungrouped)
    print("\n3. Decoy 2: Fixing single split via LeaveOneOut CV (ungrouped rows):")
    preds_loo = cross_val_predict(pipe, X, y, cv=LeaveOneOut(), n_jobs=-1)
    decoy2_acc = accuracy_score(y, preds_loo)
    print(f"   Accuracy: {decoy2_acc:.4f} ({decoy2_acc*100:.2f}%)")
    print(f"   Score remains heavily inflated (~96%), proving split variance is NOT the root cause.")

    # Anatomy of the real bug: Leaked vs Un-leaked rows in the original split
    tr_groups = set(groups.loc[idx_tr])
    te_groups = groups.loc[idx_te]
    leaked_mask = te_groups.isin(tr_groups)
    preds_test = clf.predict(X_te)
    acc_leaked = accuracy_score(y_te[leaked_mask], preds_test[leaked_mask])
    acc_unleaked = accuracy_score(y_te[~leaked_mask], preds_test[~leaked_mask])
    print(f"\n4. Anatomy of the Real Bug (Group Leakage):")
    print(f"   Test set size: {len(idx_te)}")
    print(f"   Test samples with applicant in train (leaked): {leaked_mask.sum()} ({leaked_mask.mean()*100:.1f}%)")
    print(f"   -> Accuracy on leaked rows: {acc_leaked:.4f} ({acc_leaked*100:.1f}%) [1-NN matches same person!]")
    print(f"   Test samples with new applicant (un-leaked): {(~leaked_mask).sum()} ({(~leaked_mask).mean()*100:.1f}%)")
    print(f"   -> Accuracy on un-leaked rows: {acc_unleaked:.4f} ({acc_unleaked*100:.1f}%)")

    # The Real Fix: LeaveOneGroupOut by applicant identity
    print("\n5. Real Fix: Grouped Leave-One-Out Cross-Validation (LeaveOneGroupOut by applicant):")
    logo = LeaveOneGroupOut()
    preds_logo = cross_val_predict(pipe, X, y, groups=groups, cv=logo, n_jobs=-1)
    true_acc = accuracy_score(y, preds_logo)
    correct_count = (y == preds_logo).sum()
    print(f"   Correct predictions: {correct_count} / {len(y)}")
    print(f"   True Accuracy: {true_acc:.6f} ({true_acc*100:.2f}%)")

    whole_percent = round(true_acc * 100)
    flag = f"SIGMOID_{whole_percent}"
    print(f"\n[+] True performance figure: {whole_percent}%")
    print(f"[+] Flag: {flag}")

    return flag, true_acc


def main():
    data_path = PROJECT_ROOT / "challenges" / "task_estonia" / "data" / "applications.csv"
    flag, acc = run_audit(data_path)

    print("\n[+] Submitting flag via SigmoidClient...")
    try:
        from optimizations.sigmoid_api import SigmoidClient
        client = SigmoidClient()
        repo_url = "https://github.com/Helloguys3324/MML/blob/main/challenges/task_estonia/solve.py"
        res = client.submit_text("task_estonia", flag, url=repo_url)
        print(f"[+] Submission response: {res}")
    except Exception as e:
        print(f"[-] Submission failed: {e}")


if __name__ == "__main__":
    main()
