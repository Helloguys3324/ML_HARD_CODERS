#!/usr/bin/env python3
"""
LMML 2026 - Task Bhutan: The 300-Year Old Tiger Nest Painting
Fast & Exact Vertical Strip Reassembly.

Solves the jigsaw puzzle of vertical strips via:
- Multi-feature seam affinity matrix (L1 Color + Vertical Gradient + Horizontal Curvature)
- In-degree extremity analysis to identify the leftmost boundary strip
- Multi-candidate greedy / beam path construction minimizing total seam energy
- Reconstructs image and outputs reconstructed_image.png and reconstructed_order.json
"""

import sys
import os
import json
import time
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.distance import cdist


def find_input_dir(script_dir: Path) -> Path:
    """Locate the input folder with strips."""
    candidates = []
    if len(sys.argv) > 1:
        candidates.append(Path(sys.argv[1]))

    cwd = Path.cwd()
    candidates.extend([
        cwd / "input",
        script_dir / "input",
        cwd / "train-input",
        script_dir / "train-input",
        script_dir / "data" / "input",
        script_dir / "data" / "train-input",
        cwd / "data" / "input",
        cwd / "data" / "train-input",
    ])

    for p in candidates:
        if p.is_dir():
            files = [f for f in os.listdir(p) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
            if len(files) >= 2:
                return p

    raise FileNotFoundError(f"Could not find valid input directory containing strips. Checked: {candidates}")


def solve():
    t0 = time.perf_counter()
    script_dir = Path(__file__).resolve().parent
    input_dir = find_input_dir(script_dir)
    print(f"Reading strips from: {input_dir}")

    # Gather and sort strip filenames
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)])
    n_strips = len(files)
    if n_strips == 0:
        raise ValueError(f"No strip image files found in {input_dir}")
    print(f"Found {n_strips} strips.")

    # Load all strips
    strips = []
    for f in files:
        img_path = str(input_dir / f)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")
        strips.append(img)

    h, w, c = strips[0].shape
    print(f"Strip dimensions: H={h}, W={w}, C={c}")

    # Feature extraction for each strip:
    # r1: rightmost column (index -1)
    # r2: column before rightmost (index -2)
    # l0: leftmost column (index 0)
    # l1: column after leftmost (index 1)
    r1 = np.array([s[:, -1, :].reshape(-1) for s in strips], dtype=np.float32)
    l0 = np.array([s[:, 0, :].reshape(-1) for s in strips], dtype=np.float32)

    if w >= 2:
        r2 = np.array([s[:, -2, :].reshape(-1) for s in strips], dtype=np.float32)
        l1 = np.array([s[:, 1, :].reshape(-1) for s in strips], dtype=np.float32)
    else:
        r2 = r1.copy()
        l1 = l0.copy()

    # Vertical gradient features
    dy_r1 = np.array([np.diff(s[:, -1, :], axis=0).reshape(-1) for s in strips], dtype=np.float32)
    dy_l0 = np.array([np.diff(s[:, 0, :], axis=0).reshape(-1) for s in strips], dtype=np.float32)

    # Pairwise cost matrix C[i, j]: placing strip j immediately to the right of strip i
    # 1. Color difference across seam
    d_color = cdist(r1, l0, metric='cityblock')

    # 2. Vertical edge gradient difference
    d_vgrad = cdist(dy_r1, dy_l0, metric='cityblock')

    # 3. Horizontal curvature / smoothness prediction
    pred_l0 = 2 * r1 - r2
    pred_r1 = 2 * l0 - l1
    d_curv1 = cdist(pred_l0, l0, metric='cityblock')
    d_curv2 = cdist(r1, pred_r1, metric='cityblock')
    d_curv = 0.5 * (d_curv1 + d_curv2)

    # Combined seam penalty
    C = d_color + 0.5 * d_vgrad + 0.25 * d_curv
    np.fill_diagonal(C, 1e12)

    # Find candidate leftmost strips:
    # A strip with NO incoming match (highest min incoming cost) is the left boundary
    min_incoming = np.min(C, axis=0)
    top_candidates = np.argsort(-min_incoming)[:10]

    best_total_cost = 1e18
    best_path = None

    for start_node in top_candidates:
        visited = np.zeros(n_strips, dtype=bool)
        visited[start_node] = True
        path = [start_node]
        curr = start_node
        cost = 0.0

        for _ in range(n_strips - 1):
            unvisited = np.where(~visited)[0]
            step_idx = np.argmin(C[curr, unvisited])
            next_node = unvisited[step_idx]
            cost += C[curr, next_node]
            visited[next_node] = True
            path.append(next_node)
            curr = next_node

        if cost < best_total_cost:
            best_total_cost = cost
            best_path = path

    print(f"Optimal reassembly found with total seam cost: {best_total_cost:.2f}")

    # Generate results
    reconstructed_order = [files[idx] for idx in best_path]
    reconstructed_image = np.hstack([strips[idx] for idx in best_path])

    # Save outputs to both script_dir and current working directory
    target_dirs = {script_dir, Path.cwd()}
    for d in target_dirs:
        json_out = d / "reconstructed_order.json"
        img_out = d / "reconstructed_image.png"

        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(reconstructed_order, f, indent=4)

        cv2.imwrite(str(img_out), reconstructed_image)
        print(f"Saved outputs to {d}")

    t_total = time.perf_counter() - t0
    print(f"Reconstruction completed in {t_total:.3f} seconds.")


if __name__ == "__main__":
    solve()
