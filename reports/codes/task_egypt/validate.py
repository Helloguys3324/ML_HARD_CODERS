#!/usr/bin/env python3
"""Validate main.py against the provided sample and extra synthetic cases."""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SOL = Path(__file__).resolve().parent
DATA = SOL / "data" / "LMML_Egypt_TRAIN_DATA"
PY = "/home/kozak/mllm/py"


def run_case(name, inp_lines, expect=None):
    d = Path(tempfile.mkdtemp(prefix="egypt_"))
    shutil.copy(SOL / "main.py", d / "main.py")
    (d / "test-input.txt").write_text("\n".join(inp_lines) + "\n", encoding="utf-8")
    # run from an unrelated cwd to prove path discovery
    proc = subprocess.run(
        [PY, str(d / "main.py")],
        cwd="/tmp",
        capture_output=True,
        text=True,
        timeout=60,
    )
    ok = True
    if proc.returncode != 0:
        print(f"[{name}] FAIL rc={proc.returncode}\n{proc.stderr}")
        return False
    outp = d / "test-output.txt"
    if not outp.is_file():
        print(f"[{name}] FAIL: no output file")
        return False
    out_lines = outp.read_text(encoding="utf-8").splitlines()
    if len(out_lines) != len(inp_lines):
        print(f"[{name}] FAIL: {len(out_lines)} output lines for {len(inp_lines)} inputs")
        ok = False
    # Also verify the stricter interpretation: plain split('\n') must give
    # exactly one element per input record (i.e. no trailing blank line).
    raw_lines = outp.read_text(encoding="utf-8").split("\n")
    if len(raw_lines) != len(inp_lines):
        print(f"[{name}] FAIL: raw split('\\n') gives {len(raw_lines)} items for {len(inp_lines)} inputs")
        ok = False
    parsed = []
    for i, ln in enumerate(out_lines):
        try:
            parsed.append(json.loads(ln))
        except Exception as e:
            print(f"[{name}] FAIL line {i+1} invalid JSON: {e}: {ln!r}")
            ok = False
            parsed.append(None)
    if expect is not None:
        for i, (got, want) in enumerate(zip(parsed, expect), 1):
            if got != want:
                print(f"[{name}] FAIL line {i}: got {got!r} want {want!r}")
                ok = False
    print(f"[{name}] {'OK' if ok else 'FAIL'} ({len(out_lines)} lines)")
    shutil.rmtree(d, ignore_errors=True)
    return ok


def main():
    ok = True
    # 1. Provided sample, exact expected parsed data.
    inp = (DATA / "train-input.txt").read_text(encoding="utf-8").splitlines()
    exp = [json.loads(x) for x in (DATA / "train-output.txt").read_text(encoding="utf-8").splitlines()]
    ok &= run_case("sample", inp, exp)

    # 2. Spec example.
    ok &= run_case(
        "spec-example",
        ['page 22| king :Menes| title: ruled with "the Aten" |[torn]| alive :true | Chapter IX'],
        [{"king": "Menes", "title": 'ruled with "the Aten"', "alive": True}],
    )

    # 3. Edge cases: typing, escaping, first-colon-only, noise with colons.
    cases = [
        'a: | b:  true | c : false | d: 40 | e: 0 | f: -3 | g: 3:1 | h: C:\\x\\y | i: "q"',
        'ratio: 3:1|weird: http://x.y/z|Chapter IX: The Fall|page 22: 33|###|~~~|(margin)|[torn]',
        'x: 3.14|y: 007|z: +5|w: 1e3|v: TRUE|u: False|t: null|s: -0',
        'k:  spacey  value  | k2: a|b? | k3: 2024-01-02 | k4: 50% |k5:\u00e9\u00e9\u00e9',
        '  alone :  ',
        '',
    ]
    ok &= run_case("edge", cases)
    print("ALL OK" if ok else "SOME FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
