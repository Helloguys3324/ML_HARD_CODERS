"""Adversarial same-family tests for main_v2.py.

Each case is a hand-built TRUTH in the official family, stressing one axis:
tiny/huge active cap, gate == interaction pair, 4-variable gate, tiny jump,
tiny linear coefficients, zero linear coefficients on interaction/gate dials,
strong interaction, etc.  Scored with the real evaluation_script.
"""
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

NAMES = ["x1", "x2", "x3", "x4", "x5"]


def truth(linear, pair, coef, gate_vars, gate_coef, thr, jump,
          intercept=1.0, form="centered_product"):
    rel = set(v for v in NAMES if linear.get(v, 0.0) != 0.0)
    rel.update(pair)
    rel.update(gate_vars)
    return {
        "intercept": intercept,
        "linear": {v: float(linear.get(v, 0.0)) for v in NAMES},
        "interaction": {"variables": tuple(pair), "form": form,
                        "coefficient": float(coef), "center": 0.5},
        "discontinuity": {
            "gate_variables": tuple(gate_vars),
            "boundary": {v: float(gate_coef.get(v, 0.0)) for v in NAMES},
            "threshold": float(thr), "jump": float(jump),
            "active_when": "greater_equal",
        },
        "relevant": [v for v in NAMES if v in rel],
        "irrelevant": [v for v in NAMES if v not in rel],
    }


CASES = {
    # tiny cap: active only at the very corner
    "tiny_cap": truth({"x1": 1.2, "x3": 0.5, "x5": -0.7},
                      ("x3", "x5"), -3.0, ("x1", "x4"), {"x1": 1, "x4": 1},
                      1.97, 1.4),
    # huge cap: almost everything is active
    "huge_cap": truth({"x1": -1.1, "x2": 0.9, "x4": 0.4},
                      ("x2", "x4"), 2.5, ("x1", "x5"), {"x1": -1, "x5": 1},
                      -0.65, 0.8),
    # gate pair == interaction pair
    "same_pair": truth({"x1": 0.6, "x2": -0.9, "x3": 1.3},
                       ("x1", "x3"), -4.5, ("x1", "x3"),
                       {"x1": 1, "x3": 1}, 1.55, 1.7),
    # four-variable gate
    "gate4": truth({"x1": 0.8, "x4": -0.6}, ("x2", "x5"), 3.0,
                   ("x1", "x2", "x3", "x5"),
                   {"x1": 0.8, "x2": 0.7, "x3": 0.6, "x5": 0.5}, 1.9, 0.9),
    # tiny jump
    "tiny_jump": truth({"x1": 1.0, "x4": 0.5}, ("x2", "x4"), -2.0,
                       ("x2", "x3"), {"x2": 1, "x3": 1}, 1.4, 0.06),
    # tiny but nonzero main effects
    "tiny_lin": truth({"x1": 0.012, "x2": -0.008, "x3": 0.0,
                       "x4": 0.02, "x5": -0.015},
                      ("x3", "x5"), 1.2, ("x2", "x4"),
                      {"x2": 1, "x4": 1}, 1.2, 1.1),
    # interaction pair has zero linear effect, gate dials have zero linear
    "zero_lin": truth({"x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0, "x5": 0.0},
                      ("x2", "x4"), -2.2, ("x1", "x5"),
                      {"x1": 1, "x5": 1}, 1.3, 1.3),
    # strong interaction, gate on the remaining dials
    "strong_int": truth({"x1": 2.5, "x3": -2.0}, ("x4", "x5"), 8.0,
                        ("x1", "x2"), {"x1": 1, "x2": 1}, 1.35, 2.0),
    # raw product form (not centered)
    "raw_product": truth({"x1": 0.7, "x3": -0.4, "x4": 1.1},
                         ("x2", "x4"), 2.0, ("x3", "x5"),
                         {"x3": 1, "x5": 1}, 1.6, 0.9, form="product"),
    # gate with a tiny second weight
    "skew_gate": truth({"x1": 0.9, "x2": 0.4}, ("x2", "x3"), -2.5,
                       ("x1", "x4"), {"x1": 1.0, "x4": 0.08}, 0.95, 1.2),
    # near-axis-aligned gate (one weight dominant)
    "axis_gate": truth({"x1": 1.2, "x3": 0.6}, ("x3", "x5"), -3.3,
                       ("x2", "x4"), {"x2": 1.0, "x4": 0.02}, 0.4, 1.5),
    # all five dials live, 3-var gate
    "all_live": truth({"x1": 0.5, "x2": 0.7, "x3": -0.9, "x4": 1.0, "x5": -0.3},
                      ("x1", "x4"), 3.3, ("x2", "x3", "x5"),
                      {"x2": 0.8, "x3": 1.1, "x5": -0.6}, 0.55, 1.05),
}


def main():
    pts = json.load(open(os.path.join(DATA, "challenge", "challenge_set.json")))["points"]
    main_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "main_v2.py")
    scores = []
    for name, t in CASES.items():
        mod = LE.make_module(t)
        res, oracle, sub = LE.run_one(main_path, mod, pts, 150, verbose=False)
        got = []
        if sub and sub.get("discontinuities"):
            got = sub["discontinuities"][0].get("gate_variables", [])
        gv = list(t["discontinuity"]["gate_variables"])
        print(f"{name:12s} score={res['score'] if res else -1:6.2f} "
              f"nq={len(oracle.log):3d} gate={sorted(got)} true={sorted(gv)} "
              f"rel={sub['relevant_variables'] if sub else None}")
        if res and res["score"] < 100:
            for k, v in res["breakdown"].items():
                if v["points"] < v["max"] - 0.01:
                    print("   ", k, v["points"], v.get("detail"))
            print("    notes:", sub["notes"] if sub else None)
        scores.append(res["score"] if res else 0)
    print(f"\nmean {np.mean(scores):.2f}  full {sum(1 for s in scores if s >= 99.99)}/{len(scores)}")


if __name__ == "__main__":
    main()
