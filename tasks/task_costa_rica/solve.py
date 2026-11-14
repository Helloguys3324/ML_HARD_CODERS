"""Local evaluator for task_costa_rica redaction.

Runs main.redact over train-input.json and compares against the golden
redaction, reporting:
  - exact-match vs golden (should be ~100% if we match the reference style)
  - a PII leakage probe (how many golden-replaced spans still leak)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
inp = json.loads((DATA / "train-input.json").read_text(encoding="utf-8"))
golden = json.loads((DATA / "train-output.json").read_text(encoding="utf-8"))[
    "redacted_documents"
]

docs = inp["documents"]
exact = 0
total = len(docs)
details = []

for doc in docs:
    did = doc["id"]
    got = main.redact(doc["text"])
    g = golden.get(did, "")
    ok = got == g
    exact += int(ok)
    if not ok:
        # show a compact diff for debugging
        details.append((did, got, g))

print(f"Exact golden match: {exact}/{total}")
for did, got, g in details:
    print("=" * 70)
    print("DOC", did)
    # find first differing line
    gl = g.split("\n")
    gotl = got.split("\n")
    for i in range(max(len(gl), len(gotl))):
        a = gl[i] if i < len(gl) else "<missing>"
        b = gotl[i] if i < len(gotl) else "<missing>"
        if a != b:
            print(f"  golden[{i}]: {a!r}")
            print(f"  got   [{i}]: {b!r}")
            break