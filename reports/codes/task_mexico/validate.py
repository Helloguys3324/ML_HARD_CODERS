"""Validate main.py exactly against the labelled practice data (and stress variants)."""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/home/kozak/mllm/MML/work/tasks/task_mexico/data"


def run_case(csv_path, tag):
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(csv_path, os.path.join(tmp, "test-input.csv"))
        shutil.copy(os.path.join(HERE, "main.py"), os.path.join(tmp, "main.py"))
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        proc = subprocess.run([sys.executable, "main.py"], cwd=tmp,
                              capture_output=True, text=True, env=env, timeout=900)
        outp = os.path.join(tmp, "test-output.json")
        if proc.returncode != 0:
            print(f"[{tag}] FAIL rc={proc.returncode}\n{proc.stderr[-3000:]}")
            return None
        if not os.path.exists(outp):
            print(f"[{tag}] FAIL no test-output.json")
            return None
        with open(outp) as f:
            got = json.load(f)
        print(f"[{tag}] rc=0 pairs={len(got['matches'])} stderr={proc.stderr.strip()[-200:]}")
        return got


def compare(got, expected_path, tag):
    exp = json.load(open(expected_path))["matches"]
    A = set(frozenset(p) for p in got["matches"])
    B = set(frozenset(p) for p in exp)
    missing = B - A
    extra = A - B
    print(f"[{tag}] expected={len(B)} got={len(A)} missing={len(missing)} extra={len(extra)}")
    if missing:
        print("  missing sample:", [tuple(x) for x in list(missing)[:5]])
    if extra:
        print("  extra sample:", [tuple(x) for x in list(extra)[:5]])
    return not missing and not extra


if __name__ == "__main__":
    got = run_case(os.path.join(DATA, "train-input.csv"), "train")
    ok = got is not None and compare(got, os.path.join(DATA, "train-output.json"), "train")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
