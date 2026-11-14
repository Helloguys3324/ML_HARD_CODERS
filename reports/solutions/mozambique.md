# Solution Report: task_mozambique ("The Professor of Maputo Bay")

## 1. Overview & Objective
- **Task ID:** `task_mozambique` (Mozambique).
- **Task Type:** Turn-Based Multi-Agent Strategy / Logistics Planning under Adversarial Competition.
- **Submission Mode:** Autonomous Python agent script `main.py`.
- **Environment & Game Mechanics:**
  - Takes place over 80 rounds on a connected graph of ports in Maputo Bay between the player agent and a deterministic opponent ("The Professor").
  - Interfaced via the binary `mozambique-engine` and the `arena.py` harness over JSON-RPC / stdio.
  - Each round, the agent issues an action to one idle boat. Boat cargo capacity is 3 units.
  - Transit between ports $i$ and $j$ consumes fixed rounds `routes[i][j]`.
  - Port Roles:
    - Role `0`: Timber harvest ($T$).
    - Role `1`: Fibre harvest ($F$).
    - Role `2`: Manufacturing site / workshop (crafts finished goods, type 2).
    - Base home harbor (`homes`).
  - Economy & Production:
    - Constructing a workshop (`build_workshop`) and initiating production (`craft`) both consume 2 timber and 1 fibre stored at the factory warehouse. Production takes 3 rounds.
    - Delivery (`deliver`) fulfills contracts:
      - Raw contracts: Require 2 raw units; reward **5 points**.
      - Crafted contracts: Require 1 finished good; reward **12 points**.
    - Contracts carry expiration deadlines and expire if unfulfilled.
- **Victory Condition:**
  - Total available point pool is fixed at 68 points (4 raw @ 5 pts + 4 crafted @ 12 pts: $4 \times 5 + 4 \times 12 = 68$).
  - Winning requires outscoring the Professor ($\ge 65\%$ win rate in worst-of-5 series) to unlock the flag `SIGMOID_<WORD>`.

---

## 2. Key Insights & Methodology

1. **Point Arithmetic & Win Pathways:**
   - Because the point pool totals 68, victory is achievable via two routes:
     - **3-Crafted Strategy:** Deliver 3 finished goods ($3 \times 12 = 36$ points), leaving the Professor with at most $68 - 36 = 32$ points, guaranteeing victory ($36 > 32$).
     - **Hybrid Strategy:** Deliver 2 finished goods and at least 3 raw contracts ($2 \times 12 + 3 \times 5 = 39$ points vs. max 29).
   - The Professor executes aggressive early crafting (first delivery at rounds 18–31) and scores 39–68 points if unopposed.

2. **Algorithmic Strategy (`main.py`):**
   - **All-Pairs Shortest Paths:** Floyd–Warshall computes `dist[i][j]` and next-hop lookup `nxt[i][j]` during initialization.
   - **Optimal Factory Placement:** Selects factory node $F$ minimizing weighted transit distance:
     $$\text{Cost}(F) = \text{dist}(\text{home}, F) + 0.7 \cdot (\text{dist}_{T} + \text{dist}_{\text{fib}}) + 0.35 \cdot (\text{dist}_{C_1} + \text{dist}_{C_2})$$
   - **Coordinated Opening:** Home starts with $5T + 3F$. Both boats load $2T + 1F$ at home (4 rounds), sail simultaneously to factory $F$, unload materials, construct the workshop (`build_workshop`), and start crafting the first good (`craft`).
   - **Shuttle Supply & Delivery Loop:** Boats shuttle between raw ports and the workshop to maintain an active crafting pipeline. Once a finished good is produced, a free boat immediately delivers it to the nearest valid crafted contract with the earliest deadline.
   - **Opportunistic Denial & Deadlock Prevention:** Captures raw contracts to deny points when the craft line is buffered. Undelivered cargo is offloaded back at the factory to prevent cargo deadlocks.

---

## 3. Implementation Details

Implemented in `main.py` (17 KB) using Python standard library, fully compatible with PyPy-3.10:

1. **Agent Logic (`main.py`):**
   - `prep(obs, memory)`: Floyd–Warshall path precomputation, factory node scoring, and contract indexing (`craft_idx`, `raw_idx`).
   - `agent(obs, memory)`: Hierarchical state machine generating atomic commands: `sail`, `gather`, `load`, `unload`, `build_workshop`, `craft`, and `deliver`.
   - `serve()`: Robust JSON-RPC / stdio loop handling engine communication.

2. **Benchmarking Harnesses:**
   - `solve.py`: Multi-map evaluation runner across 24 seeds using official `arena.py`.
   - `harness.py`: In-process arena caller for profiling and bottleneck identification.
   - `eval.py`: Multi-threaded worst-of-5 tournament evaluator.
   - `agent_dev.py`, `agent_v3.py`: Incremental agent logic iterations.

---

## 4. Results & Artifacts
- **Submission Script:** `main.py` (autonomous PyPy/CPython agent).
- **Answer Marker:** `answer.txt` pointing to `FILE_SUBMISSION:main.py`.
- **Runtime Metrics:** Turn execution latency is $0.14 \dots 0.35$ s on CPython (faster on PyPy), comfortably meeting engine deadlines (`within_time_limit: true`).
- **Game Evaluation:**
  - Consistently executes the opening, builds the workshop, and delivers 2 crafted contracts (24 points).
  - Secures 36:32 victories on favorable map seeds with 3 crafted deliveries.
  - Across broader randomized series, the Professor averages 34–44 points via parallel raw contract gathering, keeping overall win rate below the 65% target (confidence: **LOW**).
- **Future Direction:** Introduce beam-search lookahead planning to actively predict and intercept the Professor's raw contract gathering routes.
