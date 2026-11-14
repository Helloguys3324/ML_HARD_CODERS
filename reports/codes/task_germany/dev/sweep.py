import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import local_eval as LE  # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    gate_sizes = json.loads(sys.argv[2]) if len(sys.argv) > 2 else [1, 2, 3]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    main_path = sys.argv[4] if len(sys.argv) > 4 else os.path.join(HERE, "main_v2.py")
    points = json.load(open(os.path.join(LE.DATA, "challenge", "challenge_set.json")))["points"]
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n):
        gs = int(rng.choice(gate_sizes))
        truth = LE.random_truth(rng, gate_size=gs)
        mod = LE.make_module(truth)
        res, oracle, sub = LE.run_one(main_path, mod, points, 150, verbose=False)
        gv = truth["discontinuity"]["gate_variables"]
        true_sup = tuple(sorted("x1 x2 x3 x4 x5".split().index(v) for v in gv))
        notes = sub["notes"] if sub else ""
        m = re.search(r"supp:([^|]+)", notes)
        cands = []
        if m:
            for part in re.finditer(r"\(([^)]*)\):([-\d.]+)", m.group(1)):
                idx = tuple(int(x.strip()) for x in part.group(1).split(",") if x.strip())
                cands.append((float(part.group(2)), idx))
        cands.sort(reverse=True)
        top = cands[0][1] if cands else None
        hit = sub.get("discontinuities") or []
        got = ()
        if hit:
            got = tuple(sorted("x1 x2 x3 x4 x5".split().index(v)
                               for v in hit[0].get("gate_variables", [])))
        rows.append(dict(k=k, gs=gs, true=true_sup, top=top, got=got,
                         score=res["score"] if res else -1,
                         nq=len(oracle.log),
                         margins={str(c[1]): round(c[0], 3) for c in cands[:6]}))
        print(json.dumps(rows[-1]))
    ok = sum(1 for r in rows if r["true"] == r["got"])
    top_ok = sum(1 for r in rows if r["true"] == r["top"])
    s100 = sum(1 for r in rows if r["score"] >= 99.9)
    mean = np.mean([r["score"] for r in rows])
    print(f"\nsupport correct {ok}/{len(rows)}; top-margin correct {top_ok}/{len(rows)}; "
          f"100-score {s100}/{len(rows)}; mean {mean:.2f}; mean queries {np.mean([r['nq'] for r in rows]):.1f}")


if __name__ == "__main__":
    main()
