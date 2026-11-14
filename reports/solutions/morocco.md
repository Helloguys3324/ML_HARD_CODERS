# Solution Report: task_morocco ("The Dispatch Desk of Fez")

## 1. Overview & Objective
- **Task ID:** `task_morocco` (Morocco, LMML 2026).
- **Score & Status:** **200 / 200 points (PASSED)**.
- **Unlocked Flag:** **`SIGMOID_ROUTING`**.
- **Task Type:** LLM Routing under Quality and Budget Constraints; submitted as an autonomous Python script `main.py`.
- **Dataset:** `test-input.csv` (`request_id, prompt`) containing 500 hidden queries. The training set contains 400 queries (`train-input.csv`) paired with ground-truth answers and outputs from both candidate models.
- **Target LLM Tiers:**
  - Cheap model: Qwen2.5-0.5B (Cost: 1 dirham/request).
  - Expensive model: Qwen2.5-7B (Cost: 25 dirhams/request).
- **Dual-Gate Passing Criteria:**
  1. **Quality:** Accuracy against ground truth must be **$\ge 0.80$** (80%).
  2. **Budget:** Total cost across 500 test queries must be **$\le 10\,000$ dirhams** (average $\le 20$ dirhams/query, permitting at most 395 calls to the expensive model).
- **Output Format:** Routing decision table `routing_log.csv` (`request_id, model`).

---

## 2. Key Insights & Methodology

1. **Model Performance Profiling (400 Training Queries):**
   | Query Type | Sample Count ($n$) | Cheap Acc | Expensive Acc | Net Gain ($\Delta = \text{exp} - \text{cheap}$) |
   |---|---|---|---|---|
   | `crates_remain` (multi-step inventory arithmetic) | 24 | 0% | 100% | **+1.00** |
   | `toll` (conditional customs duty calculations) | 66 | 9% | 97% | **+0.88** |
   | `takings_minmax` (revenue extrema lookup) | 39 | 15% | 100% | **+0.85** |
   | `time` (schedule arithmetic with layovers) | 66 | 14% | 82% | **+0.68** |
   | `board_indirect` (two-hop noticeboard search) | 22 | 55% | 100% | **+0.45** |
   | `takings_fee` ("most" — maximum fee) | 13 | 23% | 69% | **+0.46** |
   | `walk_direct` (shortest path is the direct edge) | 37 | 51% | 89% | **+0.38** |
   | `crates_total` (single-step multiplication) | 43 | 70% | 100% | **+0.30** |
   | `takings_fee` ("least" — minimum fee) | 15 | 27% | 40% | **+0.13** |
   | `walk_indirect` (shortest path requires a detour) | 30 | 3% | 10% | **+0.07** |
   | `board_direct` (direct noticeboard string lookup) | 45 | 100% | 100% | **0.00** |

2. **Dijkstra Graph Analysis Inside the Router:**
   - For walking path queries across Fez quarters, the router parses city edges and transition times directly from the prompt text and executes Dijkstra's algorithm.
   - If the direct edge is indeed the shortest route (`walk_direct`), the expensive model succeeds reliably (89%).
   - If an indirect detour is shorter (`walk_indirect`), both models almost universally fail (10% vs. 3%). Allocating 25 dirhams to indirect walking queries wastes budget for negligible gain; routing them to the cheap model preserves budget for arithmetic tasks.

3. **Optimal Knapsack Budget Allocation:**
   - Each query is assigned an expected utility value:
     $$\text{Value} = P(\text{correct} \mid \text{expensive}) - P(\text{correct} \mid \text{cheap})$$
   - Let $E$ be the number of expensive model calls:
     $$\text{Cost} = 25E + (N - E) = N + 24E \le 20N \iff E \le \left\lfloor \frac{19N}{24} \right\rfloor$$
   - For $N = 500$:
     $$E \le \left\lfloor \frac{9500}{24} \right\rfloor = 395 \implies \text{Cost} = 25 \times 395 + 105 = 9\,980 \le 10\,000$$
   - Queries are sorted by `Value` descending: top 395 queries are assigned to `expensive`, and the bottom 105 (all `board_direct` and `walk_indirect`) are routed to `cheap`.

---

## 3. Implementation Details

Implemented in `main.py` using Python's standard library (`csv`, `re`, `heapq`, `collections`):

1. **Graph Parsing & Dijkstra Engine (`_walk_is_direct_shortest`):**
   - Parses origin and destination (`from <A> to <B>`).
   - Extracts edges via regex `([A-Za-z]+)-([A-Za-z]+)\s+(\d+)\s+minutes`.
   - Computes shortest path via `heapq` Dijkstra and compares direct edge length with optimal distance.

2. **Regex Query Classifier & Priority Calibration:**
   - Categorizes queries and assigns calibrated values:
     - `crates_remain`: 1.00
     - `toll`: 0.88
     - `takings_minmax`: 0.85
     - `time`: 0.68
     - `board_indirect`: 0.46
     - `takings_fee_most`: 0.45
     - `walk_direct`: 0.38
     - `crates_total`: 0.30
     - `takings_fee_least`: 0.12
     - `walk_indirect`: 0.07
     - `board_direct`: 0.00

3. **Batch Routing & Output Emission:**
   - Ingests `test-input.csv`, sorts by priority score, labels top 395 as `expensive` and bottom 105 as `cheap`, and writes `routing_log.csv`.

---

## 4. Results & Artifacts
- **Submission Script:** `main.py` (deterministic, zero external dependencies).
- **Output Table:** `routing_log.csv` (500 decisions).
- **Official Evaluation on Hidden Test Set (500 queries):**
  - **Quality Score:** **0.8520 (85.2%)** (Passing threshold: $\ge 0.80$).
  - **Total Cost:** **9,980 dirhams** (Budget cap: $\le 10,000$; 395 expensive, 105 cheap).
  - **Score:** **200 / 200 points (PASSED)**.
  - **Flag:** **`SIGMOID_ROUTING`**.
- **Local Validation (400 Train Queries):**
  - Quality = 84.75%, Cost = 7,984 dirhams (limit: 8,000).
