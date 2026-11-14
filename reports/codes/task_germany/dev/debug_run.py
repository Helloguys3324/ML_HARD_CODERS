import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "..", "..", "tasks", "task_germany", "data"))
sys.path.insert(0, DATA)
sys.path.insert(0, HERE)
import local_eval as LE  # noqa: E402


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "main_v2"
    m = load(name)
    p = LE.load_practice()
    pts = json.load(open(os.path.join(DATA, "challenge", "challenge_set.json")))["points"]
    oracle = LE.MockOracle(p.evaluate, 150)
    sub = m.solve(oracle, pts)
    print("queries:", oracle.spent)
    print("notes:", sub["notes"])
    import evaluation_script as EV
    EV.HF = p
    res = EV.evaluate({"submission": sub, "queries": oracle.log, "error": None}, "dbg", pts, True)
    print(EV.render(res, reveal=True))


if __name__ == "__main__":
    main()
