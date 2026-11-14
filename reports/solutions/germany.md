# Solution Report: task_germany ("Black Box")

## 1. Problem Overview
- **Task & Track:** Hard (300 pts). A sensor test bench in Stuttgart is controlled by five analog inputs $x_1, \dots, x_5 \in [0, 1]$ and produces a deterministic, noise-free scalar response $y = f(x_1, \dots, x_5)$. Schematics and documentation are lost.
- **Constraints & Scoring Criteria:**
  - **Query Budget:** Strictly $\le 150$ queries to `Oracle` (the 151st query causes immediate disqualification with 0 points).
  - **Runtime Limit:** 50 seconds for `solve(oracle, points)`.
  - **Structural Identifiability Requirements:**
    1. `relevant_variables` / `irrelevant_variables`: Exact binary classification of all 5 inputs.
    2. `interactions`: Identify interacting pair $(x_p, x_q)$ and interaction weight $c$ within 15%.
    3. `discontinuities`: Identify gate variables, hyperplane normal $g$, threshold $T$, and step magnitude $J$ within 15%.
    4. `model_expression`: Valid analytical Python expression string.
    5. `predictions`: Predictions across 300 test points from `challenge_set.json`.
  - **OWAT Trap:** One-variable-at-a-time variation around the hypercube center (0.5) generates linear responses with $10^{-11}$ residual error, completely failing to detect centered interaction terms or the step discontinuity.
  - Reaching $\ge 70/100$ points on the hidden oracle unlocks flag `SIGMOID_<WORD>`.

## 2. Key Methodology & Analytical Family
The unknown response function follows the parametric family:
$$f(x) = C + \sum_{i=1}^5 a_i x_i + c \cdot (x_p - 0.5)(x_q - 0.5) + J \cdot \mathbf{1}[g \cdot x \ge T]$$

A 4-phase adaptive probing strategy strictly manages query consumption:
1. **Phase 1: Finite-Difference Stencils (64 queries):**
   - $m = 4$ Latin Hypercube Sampled base points $B_k \in [0, 1]^5$, separated from domain edges by $h = 0.05$.
   - A 16-point stencil is queried per base point: base $x$, 5 single displacements $x + h e_i$, and 10 pairwise displacements $x + h e_i + h e_j$ ($4 \times 16 = 64$ queries).
   - *Second mixed differences:* $D_{ij} = f(x + h e_i + h e_j) - f(x + h e_i) - f(x + h e_j) + f(x)$. For smooth regions, $D_{ij} \approx c \cdot h^2$ exclusively for true pair $(p, q)$. The median across uncrossed stencils ($|D_{ij}| < 0.5$) isolates $c$.
   - *First differences:* Linear gradients $a_i$ are computed by subtracting the centered interaction term:
     $$a_i = \text{median}\left(\frac{f(x + h e_i) - f(x)}{h} - c \cdot (x_q - 0.5)\right)$$
2. **Phase 2: Hypercube Corner Interrogation (32 queries):**
   - Residual $r(x) = f(x) - \text{smooth}(x)$ evaluates to $0$ or $J$.
   - All 32 corners $\{0, 1\}^5$ are queried. 2-means clustering separates inactive ($r \approx 0$) and active ($r \approx J$) states, resolving jump magnitude $J = c_{hi} - c_{lo}$ and candidate gate support variables across opposing 1D hypercube edges.
3. **Phase 3: Bisection & PCA Hyperplane Fit (44 queries):**
   - Directional boundary probing (`_probe_expand`) and 9-step bisections locate boundary coordinates between switched states.
   - Singular Value Decomposition (PCA) fits hyperplane normal unit vector $g$ and bias threshold $T$, oriented such that $g \cdot x_{\text{jump}} \ge T$.
4. **Phase 4: Intercept Calibration & Evaluation:**
   - Evaluates median intercept $C$ on points firmly in the inactive region ($g \cdot x < T - 0.005$) and evaluates 300 test points.

## 3. Query Accounting & Implementation Details
- **Budget Breakdown:**
  - Phase 1 (Stencils): $4 \times 16 = 64$ queries.
  - Phase 2 (Hypercube corners): 32 queries.
  - Phase 3 (Boundary probes & bisections): 44 queries.
  - **Total Queries:** Exactly **140 / 150** queries.
- **Code Structure:**
  - `solve.py` / `main.py`: Encapsulates `Solver` with `lhs`, `_two_means`, `_bisect`, and prediction generators.
  - `dev_test.py`: Synthetic oracle verification suites.

## 4. Results & Artifacts
- **Oracle Performance:**
  - Query Budget: **140 / 150** queries used.
  - Feature & Interaction Detection: 100% (40/40 pts).
  - Discontinuity Parameterization: 100% (30/30 pts).
  - Test Point Predictions (RMSE): 100% (30/30 pts).
  - Total Score: **100 / 100 points**.
- **Flag:** `SIGMOID_FACTORIAL`.
- **Files:** `solve.py`, `main.py`, `dev_test.py`.
