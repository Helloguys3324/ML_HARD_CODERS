# Solution Report: task_estonia ("This Data Is Perfect!")

## 1. Problem Overview
- **Task & Track:** Medium (200 pts). The Estonian Ministry of Digital Affairs created an e-Resident fraud detection model (1 = fraud, 0 = legitimate) claiming **93.42%** accuracy using a 1-NN classifier (`KNeighborsClassifier(n_neighbors=1)`) with `StandardScaler` across 6 numerical features on `applications.csv` (1,971 records).
- **Audit Objectives:**
  1. Audit `project_pipeline.py` to identify three questionable engineering decisions.
  2. Prove experimentally that two are decoys not responsible for the inflated metric.
  3. Identify the true root cause (data leakage).
  4. Fix the validation pipeline to compute unbiased Leave-One-Out / group cross-validation accuracy.
  5. Submit the flag `SIGMOID_<ACCURACY>` where `<ACCURACY>` is the true integer percentage.

## 2. Root Cause Analysis & Key Hypotheses
1. **Decoy 1: Pre-Split Feature Scaling:**
   - In the original pipeline, `StandardScaler().fit_transform(X)` was executed on the full dataset before `train_test_split`.
   - *Test:* Fitting `StandardScaler` strictly on training folds inside a `Pipeline` yielded **93.67%** accuracy (a delta of only +0.25%, exactly 1 sample out of 395). Pre-split scaling was not the cause of inflation.
2. **Decoy 2: Single Unstratified Random Split:**
   - The original code evaluated a single `train_test_split(..., test_size=0.2, random_state=1137)`.
   - *Test:* Standard row-level `LeaveOneOut()` produced **95.94%** accuracy (stratified split: 93.92%). The metric remained artificially high, ruling out random split bias.
3. **The Real Bug: Group / Identity Leakage via Dropped Identifiers:**
   - The engineers dropped identifier columns:
     ```python
     X = df.drop(columns=["submission_id", "date_of_birth", "citizenship", "label"])
     ```
   - *Leakage Anatomy:* 1,971 applications belonged to only 900 unique individuals (`date_of_birth` + `citizenship`). 1,611 submissions were repeated filings (2 to 4 per individual with constant fraud status). Re-application features were near-identical (mean intra-person Euclidean distance was 0.038 vs 3.365 between distinct people).
   - In the random test split, **78.0% of test samples (308/395)** had matching records from the same applicant in training. The 1-NN classifier achieved **100.0% (308/308)** on leaked applicants, but only **71.3% (62/87)** on genuinely unseen applicants.
- **Solution:** Group-aware validation using `LeaveOneGroupOut` grouped by `date_of_birth` + `citizenship`.

## 3. Implementation Details (`solve.py`)
- **Features (6 numerical):** `form_fill_seconds`, `doc_scan_quality`, `photo_match_score`, `email_domain_age_days`, `ip_geo_distance_km`, `address_history_years`.
- **Grouping:** `groups = df["date_of_birth"].astype(str) + "_" + df["citizenship"].astype(str)` (900 unique groups).
- **Baseline Reproduction:** Global `StandardScaler` + `train_test_split(test_size=0.2, random_state=1137)` + `KNeighborsClassifier(n_neighbors=1)` reproduced the claimed 93.42%.
- **Honest Pipeline Evaluation:**
  ```python
  pipe = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=1))])
  preds_logo = cross_val_predict(pipe, X, y, groups=groups, cv=LeaveOneGroupOut(), n_jobs=-1)
  true_acc = accuracy_score(y, preds_logo)
  ```

## 4. Results & Artifacts
- **Correct Predictions:** 1,511 / 1,971.
- **True Out-of-Group Accuracy:** `0.766616` (76.66%), rounding to **77%**.
- **Flag:** `SIGMOID_77`.
- **Files:** `solve.py`.
