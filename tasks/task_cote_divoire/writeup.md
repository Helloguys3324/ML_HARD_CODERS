# task_cote_divoire — The Abidjan Transit Ledger (writeup)

## TL;DR
Recovered one router that clears BOTH accuracy gates (old ≥ 0.85, new ≥ 0.85) by
retraining a single tiny linear classifier over the **union** of the old + new
tasks, using the public set (`train-*.json`, 250 rows) plus the given OLD-task
replay sample (`old_task_sample.json`, 40 rows) as rehearsal data.

Local self-check with the organizer comparator (`model_utils.accuracy_by_task`)
and a handcrafted held-out batch of 10 unseen paraphrases (new names/vehicles):
**old 1.00 / new 1.00**.

## The difference between the two checkpoints (forensics)
- `original_model.joblib` — trained only on the OLD task (5 transactional SQL
  template labels). Aces old (1.00) / new 0.00 (never saw new labels).
- `finetuned_model.joblib` — continued-trained on the NEW task only (5 spatial /
  window-function labels). Aces new (1.00) / old 0.00.

This is **catastrophic forgetting**: continued training overwrote every OLD
decision region, so the fine-tuned router confidently routes *"insert an order"*
into a spatial template. No backdoor / hidden class — the label space itself
changed (5 → 5 different labels). Half the classes are simply gone from each
checkpoint, which is why neither can pass both gates on its own.

## Recovery approach
1. Build a joint training set: `train-input/train-output` (250, 25/class over all
   10 labels) + `old_task_sample` (40, 8/old-class, ~29 unique phrasings) — a
   rehearsal/replay buffer that restores the old decision regions.
2. Featurize with `TfidfVectorizer` (word 1-2 grams + char_wb 3-5 grams in a
   `FeatureUnion`) so the model keys on robust tokens (`join/match`, `insert/add`,
   `delete/remove`, `update/stock`, `revenue/category` … `nearest/distance`,
   `within/radius`, `lag/previous`, `lead/next`, `rank/partition`), not memorized
   names.
3. Classify with `LinearSVC(C=1.0)` — deterministic, CPU-only, retrains in well
   under a minute.

Validation:
- 5-fold stratified CV on the joint set: **old 1.000 / new 1.000**.
- old_task_sample as holdout (train on 250 only): **40/40**.
- 10 handcrafted never-seen paraphrases: **10/10**.

## Deliverables (challenges/task_cote_divoire/)
| File | Purpose |
|---|---|
| `main.py` | self-contained submission: embeds the rehearsal data (gzip+base64), re-trains at runtime, reads `test-input.json`, writes `predictions.json`. No external checkpoint needed. |
| `solve.py` | produces `checkpoints/vectorizer.joblib` + `checkpoints/recovered_model.joblib` + `checkpoints/embedded_data.json`. |
| `build_main.py` | bakes `embedded_data.json` into `main.py`. |
| `eval_cv.py` | the cross-validation / holdout experiments. |
| `checkpoints/` | `vectorizer.joblib`, `recovered_model.joblib` (template-compatible), `embedded_data.json`. |

## Result (platform grading, first submission)
- **Submission ID:** 1079
- **Status:** `passed` · **Score:** 200 / 200
- Hidden test:`old-task accuracy 0.9733 (≥0.85) PASS` · `new-task accuracy 0.9867 (≥0.85) PASS`
- **FLAG: `SIGMOID_REHEARSAL`** (recovery = rehearsal/replay of the old task)

## Prediction target
10 labels: `OLD_JOIN_CUSTOMER_ORDERS`, `OLD_GROUPBY_REVENUE_BY_CATEGORY`,
`OLD_CRUD_INSERT_ORDER`, `OLD_CRUD_UPDATE_INVENTORY`, `OLD_CRUD_DELETE_ACCOUNT`,
`NEW_SPATIAL_NEAREST_WAREHOUSE`, `NEW_SPATIAL_WITHIN_RADIUS`,
`NEW_WINDOW_LAG_DELIVERY_DELAY`, `NEW_WINDOW_LEAD_NEXT_SHIPMENT`,
`NEW_WINDOW_PARTITION_RANK_FLEET`.