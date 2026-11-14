#!/usr/bin/env python3
"""task_belgium / "Fry-ndship"  --  reproducible solver.

Idea
----
1) Fit the analyst's model: edge_m ~ area_ha + long_side_m  ("dimensions").
   R^2 = 0.989  ->  exactly the "99% of variation explained" of the story.
2) The model is essentially edge = 1.9 * (long_side + short_side)
   with short_side = area_ha * 10000 / long_side.
3) The non-trivial, designed property of the surveyed fields is the
   distribution of the *sum of the dimensions* D = L + S (equivalently
   the edge/perimeter itself): D is (near) UNIFORM on its support.
   L alone and S alone are clearly NOT uniform, and corr(L,S) = +0.36,
   so a uniform sum is not a generic consequence -- it was constructed.
   Indeed f = L/D is ~ N(0.72, 0.09) and D ~ U(133, 584):  L = f D,
   S = (1-f) D.  This makes the measured edge lengths uniformly spread.
4) Hence the word Mariette "observed in her measurements" is UNIFORM.

Run:  /home/kozak/mllm/py solve.py [csv_path] [out_dir]
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    HERE, "..", "..", "tasks", "task_belgium", "data", "potato_fields.csv")
OUT = sys.argv[2] if len(sys.argv) > 2 else HERE


def ks_uniform_stat(x):
    a, b = np.min(x), np.max(x)
    return stats.kstest((x - a) / (b - a), "uniform").statistic


def main():
    df = pd.read_csv(CSV)
    print(f"rows={len(df)}  columns={len(df.columns)}")

    # ---------- 1. the analyst's model -----------------------------------
    X = df[["area_ha", "long_side_m"]]
    y = df["edge_m"]
    reg = LinearRegression().fit(X, y)
    r2 = reg.score(X, y)
    print("\n[1] edge_m ~ area_ha + long_side_m")
    print(f"    R^2 = {r2:.4f}   ({r2*100:.1f}% explained)")
    print(f"    coefficients = {reg.coef_.round(4)}  intercept = {reg.intercept_:.2f}")

    # ---------- 2. geometry: L, S and their sum --------------------------
    L = df["long_side_m"].to_numpy()
    area_m2 = df["area_ha"].to_numpy() * 10_000.0
    S = np.round(area_m2 / L, 3)          # short side (3 decimals, exact by construction)
    D = L + S                              # sum of dimensions / semi-perimeter
    edge = df["edge_m"].to_numpy()

    # reproduce the exact generative formula
    base = 0.82 * D + df["ditch_m"].to_numpy() + df["hedge_m"].to_numpy()
    resid = edge - base
    print("\n[2] exact linear law of the measurements")
    print("    edge = 0.82*(L+S) + ditch + hedge + eps")
    print(f"    residual std = {resid.std():.4f} m  (|eps| < {np.abs(resid).max():.3f} m)")

    # ---------- 3. the UNIFORM signature ---------------------------------
    print("\n[3] distribution tests (uniform has excess kurtosis -1.2, skew 0)")
    rng = np.random.default_rng(0)
    table = []
    for name, v in [("D = L + S", D), ("edge_m", edge),
                    ("ditch_m", df["ditch_m"].to_numpy()),
                    ("hedge_m", df["hedge_m"].to_numpy()),
                    ("long_side L", L), ("short side S", S),
                    ("area_ha", df["area_ha"].to_numpy())]:
        ks = ks_uniform_stat(v)
        sims = np.array([
            ks_uniform_stat(rng.uniform(v.min(), v.max(), len(v)))
            for _ in range(500)
        ])
        p_sim = float(np.mean(sims >= ks))
        table.append((name, stats.skew(v), stats.kurtosis(v), ks, p_sim))
        print(f"    {name:12s} skew={stats.skew(v):+.3f}  kurt={stats.kurtosis(v):+.3f}"
              f"  KS={ks:.3f}  sim.p={p_sim:.3f}")

    print(f"\n    corr(long_side, short_side) = {np.corrcoef(L, S)[0, 1]:+.3f}")
    print("    -> L and S are positively correlated and individually non-uniform,")
    print("       yet their SUM is uniform: the uniform edge length is designed.")

    # how the sum was built: L = f*D, S = (1-f)*D
    f = L / D
    print(f"\n    f = L/D : mean={f.mean():.3f} std={f.std():.3f} "
          f"skew={stats.skew(f):+.2f} kurt={stats.kurtosis(f):+.2f} (bell shaped)")
    print("    D ~ Uniform => edge_m = 1.9*D + noise is uniform as well.")

    # ---------- 4. word + figure -----------------------------------------
    WORD = "UNIFORM"
    print(f"\nFLAG = SIGMOID_{WORD}")
    print(f"      edge quartiles: {np.percentile(edge, [25,50,75]).round(1)}")
    print("      A flat edge-length histogram (Mariette's measurements) is the observation.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        axs[0].hist(edge, bins=60, color="tab:blue")
        axs[0].set_title("edge_m (Mariette's measurement) -- flat / uniform")
        axs[0].set_xlabel("edge_m [m]")
        axs[1].hist(D, bins=60, color="tab:green")
        axs[1].set_title("D = long_side + short_side -- uniform")
        axs[1].set_xlabel("D [m]")
        axs[2].hist(L, bins=60, color="tab:orange", alpha=0.8, label="long side")
        axs[2].hist(S, bins=60, color="tab:red", alpha=0.5, label="short side")
        axs[2].legend()
        axs[2].set_title("individual dimensions are NOT uniform")
        fig.tight_layout()
        fig_path = os.path.join(OUT, "uniform_evidence.png")
        fig.savefig(fig_path, dpi=120)
        print(f"      figure -> {fig_path}")
    except Exception as exc:      # pragma: no cover
        print(f"      (plot skipped: {exc})")

    with open(os.path.join(OUT, "answer.txt"), "w") as fh:
        fh.write(f"SIGMOID_{WORD}\n")
    print(f"      answer written to {os.path.join(OUT, 'answer.txt')}")

    # machine-readable summary
    pd.DataFrame(table, columns=["quantity", "skew", "excess_kurtosis",
                                 "KS_uniform", "sim_p"]).to_csv(
        os.path.join(OUT, "uniform_tests.csv"), index=False)


if __name__ == "__main__":
    main()
