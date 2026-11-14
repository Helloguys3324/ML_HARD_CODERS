#!/usr/bin/env python3
"""
Solution for task_belgium (Fry-ndship) - LMML 2026
Flag: SIGMOID_SUPERVISION
Antarctica Slot 3 Clue Word: belongs
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from optimizations.sigmoid_api import SigmoidClient


def main():
    data_path = Path(__file__).resolve().parent / "data" / "potato_fields.csv"
    output_dir = Path(__file__).resolve().parent

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows.")

    # Geometric perimeter from farmer's declaration
    area_m2 = df["area_ha"] * 10000.0
    short_side_m = area_m2 / df["long_side_m"]
    perim = 2.0 * (df["long_side_m"] + short_side_m)
    df["perim"] = perim

    # 1. Edge measurements residual (Mariette measured on site)
    # edge_m ~ 0.95 * perim
    res_edge = df["edge_m"] - 0.95 * perim
    df["res_edge"] = res_edge

    # 2. Hedge measurements residual (Mariette noted on site)
    # hedge_m ~ 0.04 * perim
    res_hedge = df["hedge_m"] - 0.04 * perim
    df["res_hedge"] = res_hedge

    # Plot 1: REGRESSION in edge_m residuals
    plt.figure(figsize=(20, 6))
    plt.scatter(df["perim"], res_edge, s=6, c="#1f77b4", alpha=0.8)
    plt.title("Mariette's Edge Measurements vs Perimeter: 'REGRESSION'", fontsize=16)
    plt.xlabel("Perimeter (m)", fontsize=13)
    plt.ylabel("Residual (m)", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    edge_plot_path = output_dir / "res_edge_regression.png"
    plt.savefig(edge_plot_path, dpi=300)
    plt.close()
    print(f"Saved edge plot to {edge_plot_path}")

    # Plot 2: SUPERVISION in hedge_m residuals (Mariette's site observations)
    plt.figure(figsize=(20, 6))
    plt.scatter(df["perim"], res_hedge, s=6, c="#d62728", alpha=0.8)
    plt.title("Mariette's Hedge Observations vs Perimeter: 'SUPERVISION'", fontsize=16)
    plt.xlabel("Perimeter (m)", fontsize=13)
    plt.ylabel("Residual (m)", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    hedge_plot_path = output_dir / "res_hedge_supervision.png"
    plt.savefig(hedge_plot_path, dpi=300)
    plt.close()
    print(f"Saved hedge plot to {hedge_plot_path}")

    word = "SUPERVISION"
    flag = f"SIGMOID_{word}"
    print(f"\nExtracted Word: {word}")
    print(f"Flag: {flag}")

    # Submit or verify
    client = SigmoidClient()
    repo_url = "https://github.com/Helloguys3324/MML/blob/main/challenges/task_belgium/solve.py"
    res = client.submit_text("task_belgium", flag, url=repo_url)
    print(f"Submission response: {res}")

    # Check unlocked Antarctica clue
    clues = client.get_clue_passphrase()
    slot3 = [s for s in clues.get("slots", []) if s.get("taskId") == "task_belgium"]
    if slot3:
        print(f"Antarctica Clue Slot 3: {slot3[0]}")

    return flag, res


if __name__ == "__main__":
    main()
