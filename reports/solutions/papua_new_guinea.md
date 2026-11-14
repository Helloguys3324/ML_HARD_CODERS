# Solution Report: task_papua_new_guinea ("Bihainim Angkol / Follow Your Uncle")

## 1. Overview & Objective
- **Task ID:** `task_papua_new_guinea` (Papua New Guinea).
- **Difficulty & Score:** Medium, 200 / 200 points.
- **Domain:** Reinforcement Learning / Finite-Horizon Markov Decision Processes (MDP).
- **Output Format:** Standalone policy solver `main.py` producing `test-output.csv`.
- **Premise:**
  - An agent controls a traditional fishing canoe on a $6 \times 7$ grid in Kimbe Bay over a 24-step day.
  - The local fisherman ("angkol") historically averaged **2.49 fish/day** (over 400 logged days) fishing safely in nearshore seagrass.
  - The goal is to design an autonomous policy in `main.py` achieving at least **$1.5\times$** the uncle's yield ($\ge 3.735\text{ fish/day}$) on hidden test evaluations.

---

## 2. Environment Mechanics & Problem Formulation
- **Lagoon Grid Layout (`lagoon.py`):**
  - **Home Beach `H` at $(5, 2)$:** Landed catch is counted only upon reaching `H` (empties hold). Unlanded fish in the hold at step 24 are lost.
  - **Seagrass `s`:** Safe shallow feeding grounds with low catch probability ($p_{\text{bite}} = 0.20$).
  - **Outer Reef `T` at $(0, 4)$ and $(0, 5)$:** High-yield pelagic waters ($p_{\text{bite}} = 0.90$).
  - **Barrier Reef `#` & Channel `~` at $(1, 4)$:** Impassable reef `#` isolates the open ocean except for single channel `~`.
  - **Tidal Current:** Ending a step on `~` triggers a southward tidal surge with probability $p_{\text{tide}} = 0.40$, washing the canoe back into the lagoon.
  - **Capacity Limit:** `HOLD_CAPACITY = 6` fish.
- **MDP State Space & Backward Induction:**
  - The system is a Finite-Horizon Stochastic MDP with known transition probabilities.
  - State tuple: $s = (\text{row}, \text{col}, \text{hold}, \text{steps\_left})$, where $\text{row} \in [0..5]$, $\text{col} \in [0..6]$ (36 water cells), $\text{hold} \in [0..6]$ (7 levels), and $\text{steps\_left} \in [1..24]$.
  - Total discrete state space cardinality:
    $$|\mathcal{S}| = 36 \times 7 \times 24 = \mathbf{6{,}048}\text{ states}$$
  - Rather than training approximate RL policies (PPO/DQN) subject to sampling variance, the globally optimal policy is computed exactly via **Backward Value Iteration**.

---

## 3. Implementation Details
The solver pipeline is divided between `solve.py` (exact DP solver), `policy_table.csv`, and `main.py` (self-contained evaluation runtime):

1. **Exact Transition Modeling (`solve.py`):**
   - Accurately mirrors `lagoon._advance` transition distributions across all 6 discrete actions (`NORTH`, `SOUTH`, `EAST`, `WEST`, `FISH`, `WAIT`).
   - Models the sequential resolution of fishing outcome, intended movement, and stochastic tidal push.
2. **Bellman Backward Recursion:**
   - Base case: $V_0(s) = 0$.
   - For remaining steps $k = 1, \dots, 24$, solves for the optimal action-value function:
     $$Q_k(s, a) = \mathbb{E}_{\text{bite}, \text{tide}} \left[ \Delta \text{landed} + V_{k-1}(s') \right]$$
     $$V_k(s) = \max_{a \in \mathcal{A}} Q_k(s, a), \quad \pi^*(s) = \arg\max_{a \in \mathcal{A}} Q_k(s, a)$$
3. **Packaging Autonomous Runtime (`main.py`):**
   - Precomputes all 6,048 state-action mappings and serializes them into `policy_table.csv`.
   - Embeds a zlib-compressed base64 policy payload directly inside `main.py`.
   - `main.py` executes in milliseconds with zero third-party dependencies, reading `test-input.csv`, querying the dictionary in $O(1)$, and outputting decisions to `test-output.csv`.

---

## 4. Results & Artifacts
- **Status:** PASSED
- **Score:** 200 / 200 points
- **Recovered Flag:** `SIGMOID_ELIGIBILITY`
- **Validation Performance Metrics:**
  - **Agent Expected Yield:** **5.968 fish/day**;
  - **Uncle Benchmark:** **2.558 fish/day**;
  - **Outperformance Ratio:** **2.333×** (surpassing the $1.50\times$ threshold).
- **Key Artifacts:**
  - `policy_table.csv` — full 6,048-entry optimal lookup table;
  - `main.py` — production agent for the test harness;
  - `test-output.csv` — verification trajectories.
