# Solution Report: task_bhutan ("The 300-Year Old Tiger Nest Painting")

## 1. Overview & Objective
- **Task ID:** `task_bhutan` (Bhutan).
- **Difficulty & Score:** Medium, 200 / 200 points.
- **Track:** Computer Vision, 1D Jigsaw Puzzle Reconstruction, Combinatorial Optimization.
- **Context:** Restoration of an ancient 300-year-old silk scroll painting depicting the Paro Taktsang ("Tiger's Nest") monastery in Bhutan, sliced vertically into 192 unordered strips.
- **Input Data:** Directory of 192 scrambled image strips (`input/` or `train-input/`) named arbitrarily without positional hints.
- **Verification Gates (Private Hidden Image):**
  1. **SSIM $\ge 0.90$** (Structural Similarity Index against ground truth);
  2. **Order Match Ratio $\ge 0.90$** (fraction of correctly ordered strips);
  3. **Runtime $\le 20.0$ seconds** (hard execution cap).
- **Deliverables:**
  - `main.py` (autonomous puzzle solver);
  - `reconstructed_image.png` (horizontally stitched image);
  - `reconstructed_order.json` (ordered list of strip filenames).
- **Flag Format:** `SIGMOID_<WORD>`.

---

## 2. Key Insights & Methodology

The problem maps to finding a minimum-weight Hamiltonian path over a directed seam affinity graph $G = (V, E)$:
1. **Multi-Aspect Boundary Cost Matrix $C[i, j]$:**
   Placing strip $i$ to the left of strip $j$ is evaluated across three complementary continuity signals:
   - **Color Boundary Discontinuity ($d_{\text{color}}$):** Manhattan (L1) distance between the right edge $r_1$ of strip $i$ and left edge $l_0$ of strip $j$.
   - **Vertical Edge Alignment ($d_{\text{vgrad}}$):** L1 difference between vertical gradients $\Delta_y r_1$ and $\Delta_y l_0$, enforcing continuity of architectural and cliff slopes.
   - **Second-Order Curvature Continuity ($d_{\text{curv}}$):** Linear extrapolations $\text{pred}_{l_0} = 2r_1 - r_2$ and $\text{pred}_{r_1} = 2l_0 - l_1$ evaluating curvature smoothness across the seam.
   $$C[i, j] = d_{\text{color}} + 0.5 \times d_{\text{vgrad}} + 0.25 \times d_{\text{curv}}$$
2. **Left-Edge Extremity Detection:**
   Interior strips have a tight match to their preceding strip ($\min_i C[i, j]$ is small). The absolute left boundary strip lacks a predecessor, maximizing its minimum incoming seam cost.
3. **Greedy Trajectory Search:**
   Greedy paths are generated starting from the top 10 candidates with the highest minimum incoming costs. The path minimizing the aggregate traversal cost across all 192 strips is selected.
4. **Vectorized Efficiency:**
   Implemented via `scipy.spatial.distance.cdist` and `numpy`, executing in $< 1.0$ second.

---

## 3. Step-by-Step Implementation Details

### Implementation in `main.py`
1. **Boundary Feature Extraction:**
   Extracts boundary slices for each strip $s \in \mathbb{R}^{H \times W \times C}$:
   - Left columns: $l_0 = s[:, 0, :]$, $l_1 = s[:, 1, :]$;
   - Right columns: $r_1 = s[:, -1, :]$, $r_2 = s[:, -2, :]$;
   - Gradients: $\Delta_y l_0 = \text{diff}(l_0, \text{axis}=0)$, $\Delta_y r_1 = \text{diff}(r_1, \text{axis}=0)$.
2. **Affinity Computation:**
   ```python
   d_color = cdist(r1, l0, metric='cityblock')
   d_vgrad = cdist(dy_r1, dy_l0, metric='cityblock')
   pred_l0 = 2 * r1 - r2
   pred_r1 = 2 * l0 - l1
   d_curv = 0.5 * (cdist(pred_l0, l0, 'cityblock') + cdist(r1, pred_r1, 'cityblock'))
   C = d_color + 0.5 * d_vgrad + 0.25 * d_curv
   np.fill_diagonal(C, 1e12)
   ```
3. **Hamiltonian Path Assembly:**
   Identifies the root node from `np.argsort(-np.min(C, axis=0))[:10]` and greedily steps through unvisited nodes.
4. **Artifact Export:**
   Outputs `reconstructed_order.json` and stitches strips via `np.hstack` to `reconstructed_image.png`.

---

## 4. Results & Artifacts

### Official Evaluation Metrics

| Metric Gate | Achieved Score | Required Threshold | Status |
|:---|:---:|:---:|:---:|
| **SSIM** | **1.0000** | $\ge 0.90$ | ✅ PASS |
| **Order Match Ratio** | **1.0** (192 / 192 strips) | $\ge 0.90$ | ✅ PASS |
| **Execution Time** | **< 1.0 s** | $\le 20.0$ s | ✅ PASS |
| **Overall Result** | **PASSED (200 / 200 pts)** | — | ✅ Accepted |

- **Submission Attempt:** Attempt 1 (`2026-11-14T10:31:49.012Z`, submitter: `Kozak`).
- **Flag:**
  ```text
  SIGMOID_STITCHING
  ```
- **Key Artifacts:**
  - `main.py` (5,709 bytes, standalone solver);
  - `reconstructed_order.json` (4,227 bytes, ordering manifest).
