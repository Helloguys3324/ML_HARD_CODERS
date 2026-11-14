"""task_cote_divoire — "The Abidjan Transit Ledger" recovery solver.

Problem: fine-tuning a small shared text-to-SQL router on a NEW task (5 spatial /
window-function templates) caused catastrophic forgetting of the OLD task (5
transactional templates). Neither the original checkpoint (old 1.00 / new 0.00)
nor the fine-tuned one (old 0.00 / new 1.00) clears both accuracy gates (>= 0.85
on each task independently).

Recovery: retrain ONE linear classifier on the UNION of both tasks, using the
public set (dataset/train-*.json, 250 rows covering both tasks) plus the given
OLD-task replay sample (dataset/old_task_sample.json, 40 rows) as rehearsal data.
The model is tiny and CPU-only; a full retrain takes well under a minute.

Outputs (all written under checkpoints/):
  - vectorizer.joblib     : fitted FeatureUnion(word tfidf + char-wb tfidf)
  - recovered_model.joblib: fitted LinearSVC, the recovered checkpoint
  - embedded_data.json    : the exact labeled dataset baked into main.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CKPT = HERE / "checkpoints"

OLD_LABELS = [
    "OLD_JOIN_CUSTOMER_ORDERS",
    "OLD_GROUPBY_REVENUE_BY_CATEGORY",
    "OLD_CRUD_INSERT_ORDER",
    "OLD_CRUD_UPDATE_INVENTORY",
    "OLD_CRUD_DELETE_ACCOUNT",
]
NEW_LABELS = [
    "NEW_SPATIAL_NEAREST_WAREHOUSE",
    "NEW_SPATIAL_WITHIN_RADIUS",
    "NEW_WINDOW_LAG_DELIVERY_DELAY",
    "NEW_WINDOW_LEAD_NEXT_SHIPMENT",
    "NEW_WINDOW_PARTITION_RANK_FLEET",
]
ALL_LABELS = OLD_LABELS + NEW_LABELS


def load_data() -> tuple[list[str], list[str]]:
    """Return texts + labels, de-duplicating the replay sample against the public set."""
    train_in = json.load(open(DATA / "train-input.json", encoding="utf-8"))
    train_out = {r["id"]: r["label"] for r in json.load(open(DATA / "train-output.json", encoding="utf-8"))}
    old_sample = json.load(open(DATA / "old_task_sample.json", encoding="utf-8"))

    seen: set[str] = set()
    texts: list[str] = []
    labels: list[str] = []

    for r in train_in:
        seen.add(r["text"])
        texts.append(r["text"])
        labels.append(train_out[r["id"]])

    for r in old_sample:
        if r["text"] in seen:
            continue
        seen.add(r["text"])
        texts.append(r["text"])
        labels.append(r["label"])

    return texts, labels


def build_pipeline() -> Pipeline:
    """word 1-2 gram + char_wb 3-5 gram tfidf, joined, classified by LinearSVC."""
    word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    feats = FeatureUnion([("word", word), ("char", char)])
    return Pipeline([("feats", feats), ("clf", LinearSVC(C=1.0, random_state=0))])


def main() -> None:
    CKPT.mkdir(parents=True, exist_ok=True)
    texts, labels = load_data()
    print(f"loaded {len(texts)} unique examples "
          f"({sum(1 for l in labels if l in OLD_LABELS)} old / "
          f"{sum(1 for l in labels if l in NEW_LABELS)} new)")

    pipe = build_pipeline()
    pipe.fit(texts, labels)

    feats: FeatureUnion = pipe.named_steps["feats"]
    clf: LinearSVC = pipe.named_steps["clf"]

    # Save for the template-based model_utils.py contract:
    #   X = vectorizer.transform(texts).toarray(); model.predict(X)
    joblib.dump(feats, CKPT / "vectorizer.joblib")
    joblib.dump(clf, CKPT / "recovered_model.joblib")

    # Embed the exact labeled data main.py re-trains from at runtime (self-contained).
    embedded = [{"text": t, "label": l} for t, l in zip(texts, labels)]
    json.dump(embedded, open(CKPT / "embedded_data.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # Sanity: self-check on the full data (should be ~100%).
    pred = list(pipe.predict(texts))
    acc = sum(1 for a, b in zip(labels, pred) if a == b) / len(labels)
    print(f"self-check accuracy on {len(texts)} rows: {acc:.3f}")
    print("checkpoints written to", CKPT)


if __name__ == "__main__":
    main()