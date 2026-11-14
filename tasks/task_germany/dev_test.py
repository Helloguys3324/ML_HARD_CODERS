import sys, math, json, itertools
import numpy as np

sys.path.insert(0, "/home/kozak/mllm/challenges/task_germany")
import solve as S
import importlib.util

# --- fake oracle replicating harness semantics ---
class FakeOracle:
    def __init__(self, fn):
        self.fn = fn
        self.spent = 0
        self.cache = {}
        self.budget = 150
    @property
    def remaining(self):
        return self.budget - self.spent
    @staticmethod
    def key(x):
        return tuple(round(float(v), 9) for v in x)
    def ask_many(self, pts, phase=None):
        out = []
        for p in pts:
            k = self.key(p)
            if k not in self.cache:
                if self.spent >= self.budget:
                    raise RuntimeError("OVER BUDGET")
                self.cache[k] = round(self.fn(list(map(float,p))), 6)
                self.spent += 1
            out.append(self.cache[k])
        return out
    def ask(self, x, phase=None):
        return self.ask_many([x], phase)[0]

# --- practice function (mirror of practice_function.py) ---
def practice_f(x):
    x1,_x2,x3,x4,x5 = x
    y = 2.0 + 1.5*x1 + 0.4*x3 - 0.6*x5 - 4.0*(x3-0.5)*(x5-0.5)
    if (x1+x4)>=1.7: y += 1.5
    return y

# --- synthetic family generator ---
def make_family(rng):
    # pick 3 relevant (linear) vars among 5 as subset for variety; plus interaction pair and gate set
    while True:
        lin = sorted(rng.choice(5, size=rng.integers(2,5), replace=False))
        # interaction pair: two distinct vars (may overlap lin)
        pair = rng.choice(5, size=2, replace=False)
        # gate vars: 1..3 vars
        ng = rng.integers(1,4)
        gate = rng.choice(5, size=ng, replace=False)
        # ensure at least one var is inert? usually yes; force with prob
        if rng.random() < 0.9:
            used = set(lin) | set(pair.tolist()) | set(gate.tolist())
            if len(used) >= 5:
                continue
        break
    a = np.zeros(5)
    for j in lin: a[j] = rng.choice([-1,1])*rng.uniform(0.5,1.6)
    C0 = rng.uniform(-1,2)
    c = rng.choice([-1,1])*rng.uniform(1.0,5.0)
    p,q = pair
    gate = gate.tolist()
    gv = np.zeros(5)
    for j in gate: gv[j] = rng.choice([-1,1])*rng.uniform(0.6,1.6)
    # threshold so step region is nontrivial: between min and max of g.x on box
    lo = sum(min(0, gv[j]) for j in range(5))
    hi = sum(max(0, gv[j]) for j in range(5))
    T = rng.uniform(lo+0.15*(hi-lo), hi-0.15*(hi-lo))
    J = rng.uniform(0.7, 2.5)          # jump is always positive in the real family
    def f(x):
        x=np.asarray(x,float)
        y = C0 + a@x + c*(x[p]-0.5)*(x[q]-0.5)
        if gv@x >= T: y += J
        return y
    return dict(lin=lin, pair=pair, gate=gate, a=a, C0=C0, c=c, gv=gv, T=T, J=J, f=f)

def check_result(res, truth, points):
    ok = {}
    # variables
    rel = set(res["relevant_variables"]); irr = set(res["irrelevant_variables"])
    true_rel = set(S.VARS[j] for j in truth["lin"]) | set(S.VARS[p] for p in truth["pair"].tolist()) | set(S.VARS[j] for j in truth["gate"])
    var_ok = all((v in rel) == (v in true_rel) for v in S.VARS) and \
             all((v in irr) == (v not in true_rel) for v in S.VARS)
    # interaction
    pair = truth["pair"].tolist()
    inter = [e for e in res["interactions"] if set(e["variables"])==set([S.VARS[pair[0]],S.VARS[pair[1]]])]
    inter_ok = len(inter)==1
    coef_err = None
    if inter:
        coef_err = abs(inter[0]["coefficient"]-truth["c"])/abs(truth["c"])
    # discontinuity
    disc = res["discontinuities"]
    disc_ok = False
    gate_ok=False; bound_ok=False; jump_ok=False; cos=None; ofs=None
    if disc:
        d=disc[0]
        gate_ok = set(d["gate_variables"])==set([S.VARS[j] for j in truth["gate"]])
        coefs = np.array([d["boundary"]["coefficients"].get(v,0.0) for v in S.VARS])
        T_r = d["boundary"]["threshold"]; aw = d["boundary"].get("active_when","greater_equal")
        if aw.lower() in ("less","less_equal","<=","<","below"):
            coefs=-coefs; T_r=-T_r
        n = coefs/(np.linalg.norm(coefs)+1e-15)
        gt = truth["gv"]; tn = gt/(np.linalg.norm(gt)+1e-15)
        cos = float(n@tn); Ttrue = truth["T"]/(np.linalg.norm(gt))
        ofs = abs(T_r/(np.linalg.norm(coefs)+1e-15) - Ttrue)
        jump_ok = abs(abs(d["jump"])-abs(truth["J"]))/abs(truth["J"]) < 0.15
        bound_ok = cos>=0.995 and ofs<=0.03
        disc_ok = gate_ok and bound_ok and jump_ok
    # predictions RMSE
    yt = np.array([truth["f"](p) for p in points])
    yp = np.array(res["predictions"],float)
    rmse = math.sqrt(np.mean((yp-yt)**2))
    return dict(var=var_ok, inter=inter_ok, coef_err=coef_err, disc=disc_ok,
                gate=gate_ok, bound=bound_ok, jump=jump_ok, cos=cos, ofs=ofs, rmse=rmse)

if __name__ == "__main__":
    VARS = S.VARS
    # test practice
    rng = np.random.default_rng(0)
    pts = [[rng.random() for _ in range(5)] for _ in range(300)]
    fo = FakeOracle(practice_f)
    res = S.solve(fo, pts)
    truth = dict(lin=[0,2,4], pair=np.array([2,4]), gate=[0,3], a=np.array([1.5,0,0.4,0,-0.6]),
                 C0=2.0, c=-4.0, gv=np.array([1.0,0,0,1.0,0]), T=1.7, J=1.5, f=practice_f)
    r = check_result(res, truth, pts)
    print("=== PRACTICE ===")
    print("queries:", fo.spent)
    print("relevant:", res["relevant_variables"], "irrelevant:", res["irrelevant_variables"])
    print("interactions:", res["interactions"])
    print("disc:", res["discontinuities"])
    print("expr:", res["model_expression"])
    print("metrics:", r)
    print()
    # synthetics
    npicks=40
    good_var=good_inter=good_disc=0; maxq=0; rmses=[]
    for it in range(npicks):
        truth = make_family(rng)
        fo = FakeOracle(truth["f"])
        try:
            res = S.solve(fo, [[rng.random() for _ in range(5)] for _ in range(300)])
        except Exception as e:
            print(f"[{it}] EXC {e!r}")
            continue
        r = check_result(res, truth, pts_)
        maxq=max(maxq,fo.spent)
        rmses.append(r["rmse"])
        good_var += r["var"]; good_inter += r["inter"]; good_disc += r["disc"]
        if not (r["var"] and r["inter"] and r["disc"] and r["rmse"]<0.01):
            print(f"[{it}] FAIL var={r['var']} inter={r['inter']}({r['coef_err']}) disc={r['disc']}(gate={r['gate']},bound={r['bound']}(cos={r['cos'] and round(r['cos'],5)},ofs={r['ofs'] and round(r['ofs'],5)}),jump={r['jump']}) rmse={r['rmse']:.4f} q={fo.spent}")
            print("    lin",truth['lin'],"a",np.round(truth['a'],3),"pair",truth['pair'],"c",round(truth['c'],3),"gate",truth['gate'],"gv",np.round(truth['gv'],3),"T",round(truth['T'],3),"J",round(truth['J'],3))
            print("    got rel",res["relevant_variables"],"inter",res["interactions"],"disc",res["discontinuities"])
    print(f"\n=== SYNTH ({npicks}) === var {good_var}/{npicks} inter {good_inter}/{npicks} disc {good_disc}/{npicks} maxq={maxq} rmse>0.01 count={sum(1 for r in rmses if r>0.01)}")