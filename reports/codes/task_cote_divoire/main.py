#!/usr/bin/env python3
"""task_cote_divoire - "The Abidjan Transit Ledger" recovered routing model.

Self-contained recovery of a text-to-SQL request router that forgot the OLD task
(5 transactional templates) after fine-tuning on the NEW task (5 spatial/window
templates). This script re-trains a single tiny linear classifier over the union
of BOTH tasks from an embedded rehearsal set, then speaks the exact I/O contract:

    read  test-input.json   ({"id","text"} per row)
    write predictions.json  ({"id","label"} per row)

Environment: Python 3.10-3.12, scikit-learn, numpy, joblib (all stated in the
task). A full retrain takes well under a minute on CPU.
"""

from __future__ import annotations

import base64
import gzip
import json
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

_TRAIN_B64 = "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"


def _load_train() -> tuple[list[str], list[str]]:
    raw = gzip.decompress(base64.b64decode(_TRAIN_B64)).decode("utf-8")
    rows = json.loads(raw)
    return [r["text"] for r in rows], [r["label"] for r in rows]


def _build() -> Pipeline:
    word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    feats = FeatureUnion([("word", word), ("char", char)])
    return Pipeline([("feats", feats), ("clf", LinearSVC(C=1.0, random_state=0))])


def main() -> int:
    # Same optional flags as the shipped template (filename-only, no path).
    input_path = "test-input.json"
    output_path = "predictions.json"
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] in ("--input", "-i") and i + 1 < len(args):
            input_path = args[i + 1]; i += 2
        elif args[i] in ("--output", "-o") and i + 1 < len(args):
            output_path = args[i + 1]; i += 2
        elif args[i] == "--model" and i + 1 < len(args):
            i += 2  # accepted for template-compatibility; ignored (self-contained)
        else:
            i += 1

    in_path = Path(input_path)
    if not in_path.exists():
        print(f"input file not found: {in_path}", file=sys.stderr)
        return 2

    texts, labels = _load_train()
    pipe = _build()
    pipe.fit(texts, labels)

    records = json.load(open(in_path, encoding="utf-8"))
    ids = [r["id"] for r in records]
    xt = [r["text"] for r in records]
    preds = list(pipe.predict(xt))

    out = [{"id": rid, "label": lab} for rid, lab in zip(ids, preds)]
    out_path = Path(output_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {len(out)} predictions to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
