"""Local development harness for the German 'Black Box' task.

Runs main.solve() against
  * the public practice function,
  * synthetic functions from the same family (random wiring),
  * a few adversarial variants (3-variable gate, negative gate weights,
    negative jump, nonlinear main effect, tiny/huge jump),
and scores each run exactly like evaluation_script.py does.
"""
import importlib.util
import json
import math
import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "..", "..", "tasks", "task_germany", "data"))
sys.path.insert(0, DATA)

import evaluation_script as EV  # noqa: E402


# --------------------------------------------------------------------------
class MockOracle:
    def __init__(self, f, budget=150):
        self.f = f
        self.budget = budget
        self.spent = 0
        self.cache = {}
        self.log = []

    @property
    def remaining(self):
        return self.budget - self.spent

    @staticmethod
    def key(x):
        return tuple(round(float(v), 9) for v in x)

    def ask(self, x, phase=None):
        return self.ask_many([x], phase)[0]

    def ask_many(self, points, phase=None):
        out = []
        for p in points:
            k = self.key(p)
            if k in self.cache:
                out.append(self.cache[k])
                continue
            if self.spent + 1 > self.budget:
                raise RuntimeError("OVER BUDGET")
            y = self.f([float(v) for v in p])
            self.cache[k] = y
            self.spent += 1
            self.log.append({"n": self.spent, "x": list(map(float, p)), "y": y,
                             "phase": phase})
            out.append(y)
        return out


# --------------------------------------------------------------------------
class Mod:
    pass


def make_module(truth):
    m = Mod()
    m.VAR_NAMES = ["x1", "x2", "x3", "x4", "x5"]
    m.TRUTH = truth
    lin = truth["linear"]
    it = truth["interaction"]
    d = truth["discontinuity"]
    coef = d["boundary"]
    thr = d["threshold"]
    aw = d.get("active_when", "greater_equal")
    jump = d["jump"]

    def f(x):
        x1, x2, x3, x4, x5 = x
        y = truth["intercept"] + sum(lin[v] * xi for v, xi in zip(m.VAR_NAMES, x))
        if it is not None:
            i, j = [m.VAR_NAMES.index(v) for v in it["variables"]]
            if it.get("form", "product") == "centered_product":
                c = it.get("center", 0.5)
                y += it["coefficient"] * (x[i] - c) * (x[j] - c)
            else:
                y += it["coefficient"] * x[i] * x[j]
        s = sum(coef[v] * xi for v, xi in zip(m.VAR_NAMES, x))
        act = s >= thr if str(aw).startswith("greater") else s <= thr
        if act:
            y += jump
        if "extra" in truth and truth["extra"]:
            y += truth["extra"](x)
        return y

    def gate_active(x):
        s = sum(coef[v] * xi for v, xi in zip(m.VAR_NAMES, x))
        return bool(s >= thr if str(aw).startswith("greater") else s <= thr)

    m.f = f
    m.evaluate = lambda x: round(f(x), 6)
    m.gate_active = gate_active
    m.expression = lambda: "synthetic"
    return m


def random_truth(rng, gate_size=2, jump=None, square=False):
    names = ["x1", "x2", "x3", "x4", "x5"]
    nl = int(rng.integers(0, 3))          # 0..2 inert dials
    live = list(rng.permutation(5))
    lin = {v: 0.0 for v in names}
    for i in live[: 5 - nl]:
        c = float(rng.uniform(0.4, 2.0)) * (1 if rng.random() < 0.5 else -1)
        lin[names[i]] = round(c, 3)
    cand = [i for i in live if lin[names[i]] != 0.0]
    if len(cand) < 2:
        cand = live[:]
    pair = rng.choice(np.array(cand), size=2, replace=False)
    c = float(rng.uniform(1.0, 6.0)) * (1 if rng.random() < 0.5 else -1)
    gvars = rng.choice(np.arange(5), size=gate_size, replace=False)
    coef = {v: 0.0 for v in names}
    for g in gvars:
        coef[names[g]] = round(float(rng.uniform(0.4, 1.5)), 3) * (1 if rng.random() < 0.7 else -1)
    vals = np.array([coef[v] for v in names])
    mx = float(np.maximum(vals, 0).sum())
    mn = float(np.minimum(vals, 0).sum())
    frac = float(rng.uniform(0.15, 0.85))       # position of the plane inside the cube
    thr = round(mn + frac * (mx - mn), 3)
    if jump is None:
        jump = float(rng.uniform(0.4, 2.2))
    rel = set()
    for v in names:
        if lin[v] != 0:
            rel.add(v)
    rel.add(names[pair[0]])
    rel.add(names[pair[1]])
    for g in gvars:
        rel.add(names[g])
    truth = {
        "intercept": round(float(rng.uniform(-1.0, 3.0)), 3),
        "linear": lin,
        "interaction": {"variables": (names[pair[0]], names[pair[1]]),
                        "form": "centered_product", "coefficient": round(c, 3),
                        "center": 0.5},
        "discontinuity": {"gate_variables": tuple(names[g] for g in gvars),
                          "boundary": coef, "threshold": thr, "jump": jump,
                          "active_when": "greater_equal"},
        "relevant": [v for v in names if v in rel],
        "irrelevant": [v for v in names if v not in rel],
    }
    if square:
        # extra smooth curvature (not part of the official family, stress test)
        truth["extra"] = lambda x: 0.7 * (x[2] ** 2 - 0.5 * x[2])
        truth["relevant"] = [v for v in names if v in rel or v == "x3"]
        truth["irrelevant"] = [v for v in names if v not in truth["relevant"]]
    return truth


# --------------------------------------------------------------------------
def run_one(main_path, mod, points, budget=150, verbose=True):
    spec = importlib.util.spec_from_file_location("cand_main", main_path)
    mod_main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod_main)
    oracle = MockOracle(mod.evaluate, budget)
    EV.HF = mod
    try:
        sub = mod_main.solve(oracle, points)
    except Exception:
        traceback.print_exc()
        return None, oracle, None
    out = {"submission": sub, "queries": oracle.log, "error": None}
    res = EV.evaluate(out, "dev", points, True)
    if verbose:
        print(EV.render(res, reveal=True))
    return res, oracle, sub


def load_practice():
    spec = importlib.util.spec_from_file_location("practice", os.path.join(DATA, "practice_function.py"))
    p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p)
    return p


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default=os.path.join(HERE, "..", "main.py"))
    ap.add_argument("--n-synth", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=150)
    args = ap.parse_args()
    points = json.load(open(os.path.join(DATA, "challenge", "challenge_set.json")))["points"]

    scores = []
    # practice
    print("=" * 78)
    print("PRACTICE FUNCTION")
    p = load_practice()
    res, oracle, sub = run_one(args.main, p, points, args.budget, verbose=True)
    if res:
        scores.append(("practice", res["score"], len(oracle.log)))
    # random synthetics
    rng = np.random.default_rng(args.seed)
    for t in range(args.n_synth):
        truth = random_truth(rng)
        mod = make_module(truth)
        print("=" * 78)
        print(f"SYNTH {t}: {json.dumps(truth['discontinuity']['boundary'])} "
              f"thr={truth['discontinuity']['threshold']} jump={truth['discontinuity']['jump']} "
              f"int={truth['interaction']['coefficient']} pair={truth['interaction']['variables']}")
        res, oracle, sub = run_one(args.main, mod, points, args.budget, verbose=True)
        if res:
            scores.append((f"synth{t}", res["score"], len(oracle.log)))
    print("=" * 78)
    tot = 0.0
    for name, s, nq in scores:
        print(f"{name:10s} score={s:6.2f}  queries={nq}")
        tot += s
    if scores:
        print(f"mean score = {tot/len(scores):.2f}  (n={len(scores)})")


if __name__ == "__main__":
    main()
