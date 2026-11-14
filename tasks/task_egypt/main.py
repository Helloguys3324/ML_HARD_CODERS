#!/usr/bin/env python3
"""
Always Valid, No Excuses (task_egypt) - LMML 2026
Parses raw OCR Egyptology record cards into strictly valid JSON, one line per record.
"""
import json
import re
import sys
from pathlib import Path


def parse_record(line: str) -> dict:
    record = {}
    # Each record is key: value fields separated by |
    segments = line.strip("\r\n").split("|")
    for seg in segments:
        seg = seg.strip()
        if not seg or ":" not in seg:
            continue

        # Only the first : separates key from value
        raw_key, raw_val = seg.split(":", 1)
        key = raw_key.strip()
        val = raw_val.strip()

        # Only a segment shaped like key: value with a simple word key is a real field
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", key):
            continue

        # Conversion rules for each real field (after trimming):
        # - empty value -> null
        # - exactly true / false -> JSON booleans
        # - whole number (40, 0, -3) -> JSON number
        # - anything else -> JSON string, with " and \ escaped
        if val == "":
            record[key] = None
        elif val == "true":
            record[key] = True
        elif val == "false":
            record[key] = False
        elif re.match(r"^-?(?:0|[1-9]\d*)$", val):
            record[key] = int(val)
        else:
            record[key] = val

    return record


def find_input_file(cwd: Path, script_dir: Path) -> Path:
    # 1. Command-line argument
    if len(sys.argv) > 1 and Path(sys.argv[1]).is_file():
        return Path(sys.argv[1])

    # 2. Standard expected locations
    standard_names = [
        "test-input.txt",
        "test_input.txt",
        "input.txt",
        "test.txt",
    ]
    for name in standard_names:
        if (cwd / name).is_file():
            return cwd / name
        if (script_dir / name).is_file():
            return script_dir / name

    # 3. Search directory for any candidate txt file containing '|'
    for search_dir in (cwd, script_dir):
        for p in search_dir.glob("*.txt"):
            if p.name in ("test-output.txt", "train-output.txt", "main.py.txt", "requirements.txt"):
                continue
            try:
                sample = p.read_text(encoding="utf-8", errors="ignore")[:500]
                if "|" in sample and ":" in sample:
                    return p
            except Exception:
                pass

    # 4. Fallback for local testing
    if (script_dir / "data" / "LMML_Egypt_TRAIN_DATA" / "train-input.txt").is_file():
        return script_dir / "data" / "LMML_Egypt_TRAIN_DATA" / "train-input.txt"
    if (cwd / "train-input.txt").is_file():
        return cwd / "train-input.txt"

    return Path("test-input.txt")


def main():
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()

    input_path = find_input_file(cwd, script_dir)

    lines = []
    if input_path.exists():
        with open(input_path, "r", encoding="utf-8", errors="replace") as fin:
            lines = fin.read().splitlines()

    outputs = []
    for line in lines:
        if not line.strip():
            continue
        rec = parse_record(line)
        outputs.append(json.dumps(rec, separators=(",", ":"), ensure_ascii=False))

    # Write output to test-output.txt in cwd, script_dir, and sys.argv[2] if given
    output_targets = {cwd / "test-output.txt", script_dir / "test-output.txt"}
    if len(sys.argv) > 2:
        output_targets.add(Path(sys.argv[2]))

    for out_path in output_targets:
        try:
            with open(out_path, "w", encoding="utf-8") as fout:
                for out in outputs:
                    fout.write(out + "\n")
        except Exception:
            pass


if __name__ == "__main__":
    main()
