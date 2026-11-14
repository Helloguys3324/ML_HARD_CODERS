#!/usr/bin/env python3
"""Build the self-contained main.py for task_cote_divoire.

Reads checkpoints/embedded_data.json (produced by solve.py), gzip+base64 encodes
it, and writes main.py as a fully self-contained submission: it re-trains the
recovered linear router at runtime from the embedded rehearsal data, then reads
`test-input.json` (bare filename) and writes `predictions.json` ({"id","label"}
per row). No external checkpoint/vectorizer files are required, so it runs under
either a URL-folder harness or a single-file harness.
"""

from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

TEMPLATE = r'''#!/usr/bin/env python3
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

_TRAIN_B64 = "__TRAIN_BLOB__"


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
'''


def main() -> None:
    rows = json.load(open(HERE / "checkpoints" / "embedded_data.json", encoding="utf-8"))
    raw = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    blob = base64.b64encode(gzip.compress(raw, 9)).decode("ascii")
    content = TEMPLATE.replace("__TRAIN_BLOB__", blob)
    out = HERE / "main.py"
    out.write_text(content, encoding="utf-8")
    print(f"wrote {out} ({len(content)} bytes, blob {len(blob)} chars)")


if __name__ == "__main__":
    main()