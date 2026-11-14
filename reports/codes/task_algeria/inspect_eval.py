"""Pretty-print an evaluation result JSON (from evaluation_script --out)."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(path: str) -> None:
    d = json.load(open(path, encoding="utf-8"))
    r = d["result"] if "result" in d else d
    print(json.dumps(r.get("gates", r), indent=1))
    print("passed:", r.get("passed"))
    for row in d.get("replies", []):
        flags = []
        if row.get("content_hit") is False:
            flags.append("CONTENT")
        if row.get("class_ok") is False:
            flags.append("CLASS")
        print(f"{row['id']} conf={row['conformance']:.2f} {' '.join(flags)}")
        print("  Q:", row["prompt"])
        print("  R:", row["reply"])
        print("  P:", row["plain"])
        if row.get("violations"):
            bad = [v for v in row["violations"] if v]
            if bad:
                print("  V:", bad)
    for c in d.get("calls", []):
        print(f"{c['id']} {'OK' if c['ok'] else 'BAD'} seal={c['seal'] or '?'} want={c['expected']}")
        print("  Q:", c["prompt"])
        print("  R:", c["reply"])


if __name__ == "__main__":
    main(sys.argv[1])
