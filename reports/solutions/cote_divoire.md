# Solution Report: task_cote_divoire ("The Abidjan Transit Ledger")

## 1. Overview & Objective
- **Task ID:** `task_cote_divoire` (Côte d'Ivoire).
- **Difficulty & Score:** Medium, 200 / 200 points.
- **Domain:** Natural Language Processing, Continual Learning, Catastrophic Forgetting Mitigation.
- **Context:**
  - A lightweight query router classifying user natural language intents into SQL query templates was fine-tuned on a new domain (**NEW task** — 5 spatial/window query types), inducing **catastrophic forgetting** of the baseline domain (**OLD task** — 5 transactional CRUD/join query types).
  - Baseline checkpoints exhibited total divergence:
    - `original_model.joblib`: $\text{Accuracy}_{\text{old}} = 1.00$, $\text{Accuracy}_{\text{new}} = 0.00$.
    - `finetuned_model.joblib`: $\text{Accuracy}_{\text{old}} = 0.00$, $\text{Accuracy}_{\text{new}} = 1.00$.
- **Passing Criteria:** Model must exceed 85% accuracy on both task distributions simultaneously on the hidden evaluation set:
  $$\text{Accuracy}_{\text{old}} \ge 0.85 \quad \text{and} \quad \text{Accuracy}_{\text{new}} \ge 0.85$$
- **Target Label Space (10 classes):**
  - *OLD task:* `OLD_JOIN_CUSTOMER_ORDERS`, `OLD_GROUPBY_REVENUE_BY_CATEGORY`, `OLD_CRUD_INSERT_ORDER`, `OLD_CRUD_UPDATE_INVENTORY`, `OLD_CRUD_DELETE_ACCOUNT`.
  - *NEW task:* `NEW_SPATIAL_NEAREST_WAREHOUSE`, `NEW_SPATIAL_WITHIN_RADIUS`, `NEW_WINDOW_LAG_DELIVERY_DELAY`, `NEW_WINDOW_LEAD_NEXT_SHIPMENT`, `NEW_WINDOW_PARTITION_RANK_FLEET`.
- **Submission Format:** Standalone `main.py` executing in an isolated runtime without network access, consuming `test-input.json` and producing `predictions.json`.

---

## 2. Key Insights & Methodology

### 2.1 Experience Replay / Rehearsal Buffer
- Because the label spaces of the two tasks were completely disjoint, neither pre-existing checkpoint could be deployed directly.
- The training datasets were synthesized into a unified rehearsal buffer:
  - Public dataset: `train-input.json` + `train-output.json` (250 examples, 25 per class).
  - Preserved historical replay samples: `old_task_sample.json` (40 examples, 8 per old class).
  - After text deduplication, a curated set of 279 unique annotated queries was established.

### 2.2 Dual-Granularity Feature Union & Linear SVM
- **Text Representations (`FeatureUnion`):**
  1. *Word n-grams:* `(1, 2)`-grams with sublinear term-frequency scaling.
  2. *Character n-grams:* `(3, 5)`-grams with word boundaries (`char_wb`). Captures morphological affixes and verb stems (`join`, `insert`, `delete`, `nearest`, `radius`, `lag`, `partition`) regardless of arbitrary entity names or phrasing variations.
- **Classifier:** `LinearSVC(C=1.0)` with squared hinge loss. Trains in $\approx 1.2$ seconds on CPU, maintains deterministic decision boundaries, and requires minimal memory footprint.

### 2.3 Self-Contained Deployment
The 279-sample rehearsal dataset is compressed with `gzip` and base64-encoded directly inside `main.py`. Upon container spin-up, the script decompresses the corpus into memory, fits the dual TF-IDF + `LinearSVC` pipeline in under 2 seconds, and executes inference on `test-input.json`.

---

## 3. Implementation Details

1. **`solve.py`:**
   - Merges public train sets and `old_task_sample.json`.
   - Constructs and fits the `FeatureUnion` + `LinearSVC` pipeline.
   - Saves intermediate artifacts (`vectorizer.joblib`, `recovered_model.joblib`, `embedded_data.json`).
2. **`eval_cv.py` (Validation Protocol):**
   - 5-fold `StratifiedKFold` on the combined 279-sample dataset achieved **1.000 / 1.000** accuracy across both tasks.
   - Generalization test on `old_task_sample.json` when training only on the 250 base examples scored **40 / 40 (1.00)**.
   - Paraphrased entity perturbation stress tests scored **10 / 10 (1.00)**.
3. **`build_main.py`:**
   - Bundles the compressed rehearsal dictionary and template logic into `main.py`.
4. **`main.py`:**
   - Autonomous submission script: decompresses data, trains in RAM, and outputs `predictions.json`.

---

## 4. Results & Artifacts
- **Platform Submission:** `passed` (Submission ID: 1079)
- **Score:** `200 / 200`
- **Hidden Test Accuracy:**
  - `old-task accuracy`: **0.9733** (target $\ge 0.85$) ✅
  - `new-task accuracy`: **0.9867** (target $\ge 0.85$) ✅
- **Flag:** `SIGMOID_REHEARSAL`
- **Deliverables:**
  - `main.py`: Self-contained training and inference pipeline.
  - `solve.py`: Rehearsal training script and artifact builder.
  - `eval_cv.py`: Cross-validation and robustness evaluation harness.
  - `writeup.md`: Comprehensive solution narrative.
