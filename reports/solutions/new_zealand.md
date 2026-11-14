# Solution Report: task_new_zealand ("The Braided River Run")

## 1. Overview & Objective
- **Task ID:** `task_new_zealand` (New Zealand).
- **Difficulty & Score:** Hard, 300 / 300 points.
- **Domain:** Reinforcement Learning / Deterministic Pathfinding on Graphs (A* Search).
- **Output Format:** Movement trajectory string in `solution.txt` yielding flag `SIGMOID_<NAME>`.
- **Premise:**
  - In a braided river valley mapped to a $47 \times 61$ grid, an agent must navigate from starting coordinate `S` (top-left) to the target ranger cabin `G` (bottom-right) across five treacherous river channels.
  - The park ranger holds a course record of 297 minutes. The objective is to compute an admissible, collision-free route that reaches `G` in **$\le 296$ minutes** (beating the ranger's record).
  - Movement mechanics:
    - Discrete action space: `U` (up), `D` (down), `L` (left), `R` (right), and `W` (wait).
    - Base step cost: 1 minute for movement or waiting.
    - Swamp tiles (`,`): incur an additional `swamp_cost = 3` minutes (total 4 minutes).
    - Elevation changes: climbing uphill incurs `climb_cost = 1` minute per meter of elevation gain ($\max(0, h_{\text{new}} - h_{\text{curr}})$).
    - Impassable obstacles: Rock `#` and deep water `=` cannot be traversed.
    - Tidal cycle: Ford crossings (`~`) are only passable during low tide: $t \pmod{16} < 5$.
    - Suspension bridges (`A`..`I`): All bridges start in a raised (impassable) state except bridge `I`. Walking onto winch stations (`a`..`l`) toggles the states of associated bridges via symmetric difference (XOR).

---

## 2. Key Insights & Search Methodology
- **State Space Expansion:**
  - The deterministic environment is modeled as a shortest-path problem on an expanded state graph where each node is defined by:
    $$\text{State} = (\text{cell\_index}, \text{tide\_phase}, \text{bridge\_mask})$$
    where:
    - $\text{cell\_index} \in [0, N_{\text{passable}}-1]$ ($1{,}146$ traversable cells out of $2{,}867$ grid cells);
    - $\text{tide\_phase} = t \pmod{16} \in [0, 15]$ (16 discrete phases of the tidal period);
    - $\text{bridge\_mask} \in [0, 2^8 - 1]$ (8-bit bitmask tracking dynamic bridges `A`..`H`, with bridge `I` statically open, yielding 256 configurations).
  - Total state space cardinality:
    $$|\mathcal{S}| = 1{,}146 \times 16 \times 256 \approx \mathbf{4{,}694{,}016}\text{ states}$$
- **Optimal Search via A*:**
  - Transitions depend strictly on $(\text{cell}, \text{phase}, \text{mask})$, guaranteeing deterministic edge costs and enabling exact $A^*$ search with uniform cost bounds.
- **Monotone Admissible Heuristic:**
  - Precomputes backward Breadth-First Search (BFS) distances from target cell `G` across a relaxed grid where all non-rock, non-water cells cost 1 minute (ignoring tides, bridges, swamps, and elevation).
  - This provides a strictly admissible, consistent lower bound $h(s) \le h^*(s)$ on the remaining time to `G`.

---

## 3. Implementation Details
The solver is implemented across `solve.py` (verification harness) and `main.py` (standalone production solver with zero external dependencies):

1. **Map Parsing & Graph Indexing (`parse_map` in `main.py`):**
   - Parses `test-input.txt`, extracting grid bounds, terrain tokens, elevation matrix, tide parameters (`period=16, low=5`), initial open bridges, and winch-to-bridge wiring tables.
   - Assigns unique integer indices to all traversable cells and precomputes neighbor adjacency lists.
2. **Backward BFS Heuristic Precomputation:**
   - Evaluates minimum step distances $h_{\text{dist}}[r, c]$ from `G` via `collections.deque`.
3. **High-Performance Memory Layout:**
   - To avoid dictionary allocation overhead and Python object churn over 4.7M potential states, transitions and visited distances are tracked in flat 1D C-arrays using Python's `array('i', [-1] * TOTAL_STATES)`:
     - `dist`: records minimum elapsed time to reach each expanded state;
     - `parent_state`: state index of predecessor for route backtracing;
     - `parent_action`: action character (`U`, `D`, `L`, `R`, `W`).
   - Open set is maintained using `heapq` with priority key $f = g + h$.
4. **Transition Dynamics & Edge Evaluation:**
   - Evaluates `W` (wait) and directional steps `['U', 'D', 'L', 'R']`.
   - Validates low-tide condition $(t + 1) \pmod{16} < 5$ for ford crossings `~`.
   - Checks bitmask status for bridges `A`..`H`.
   - Updates `bridge_mask` via bitwise XOR upon stepping onto winch cells `a`..`l`.
5. **Path Reconstruction:**
   - Upon popping any state located at cell `G`, the algorithm terminates immediately and reconstructs the action sequence.

---

## 4. Results & Artifacts
- **Status:** PASSED
- **Score:** 300 / 300 points
- **Recovered Flag:** `SIGMOID_BELLMAN`
- **Performance & Verification:**
  - **Total Actions:** **246 steps**;
  - **Elapsed Time:** **293 minutes** (4 minutes under the 297-minute record budget);
  - **Solver Execution Time:** $\approx 0.5\text{ seconds}$ in standard library Python.
- **Key Artifacts:**
  - `solution.txt` — verified 246-character route string;
  - `main.py` — fast standalone solver;
  - `solve.py` — research script with local simulator validation.
