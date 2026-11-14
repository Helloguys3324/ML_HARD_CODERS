# Solution Report: task_senegal ("The Teranga FM Token Diet")

## 1. Overview & Objective
- **Task ID:** `task_senegal` (Senegal).
- **Difficulty & Score:** Medium, 200 / 200 points.
- **Domain:** General AI / LLM Cost Optimization & Dynamic Query Routing.
- **Output Format:** Routing schedule `routing_config.json` and flag `SIGMOID_<NAME>`.
- **Premise:**
  - Community station Teranga FM in Kaolack (Senegal's peanut basin) operates an SMS farmer helpline backed by an LLM inference API.
  - The baseline implementation routed all incoming queries to a high-tier model with verbose styling, few-shot exemplars, and unpruned context, consuming 2,277 tokens per evaluation batch.
  - The goal is to construct an optimal routing table (`routing_config.json`) across 20 benchmark queries from `fact_bank.json` maintaining 100% QA accuracy (20/20 `PASS`) under a strict upper bound of **1,200 total tokens** (a ~49% reduction).

---

## 2. Key Insights & Routing Mechanics
- **Simulator & Verification Logic (`model_utils.py`):**
  - **Pass Criterion (`question_passes`):** A completion satisfies a question if and only if every required fact slot has at least one valid keyphrase from `slot["phrasings"]` matched as a substring.
  - **Slot Inclusion Condition (`render_completion_body`):**
    $$\text{assisted} \lor \neg \text{slot}[\text{"large\_only"}], \quad \text{where } \text{assisted} = (\text{tier} == \text{"large"}) \lor \text{context}$$
  - **Parameter Influence on Cost:**
    - `fewshot`: Injects substantial prompt overhead without altering slot eligibility $\implies$ `fewshot = False` across all calls.
    - `style`: `verbose` introduces boilerplate intro/outro phrasing; `terse` outputs compact factual text $\implies$ `style = "terse"` uniformly.
    - `context`: RAG context snippets are only required for the `cheap` tier when handling `large_only` slots. On `large`, context is redundant.
- **Optimal Routing Policy:**
  - **Questions without `large_only` slots (8 queries):** Assigned to `cheap` tier without context (`cheap / terse / fewshot=False / context=False`).
  - **Questions with `large_only` slots (12 queries):** Cost trade-off: select `cheap + context=True` if the snippet token cost is lower than the premium of `large + context=False`.

---

## 3. Implementation Details
The optimization pipeline is driven by `solve.py` and validated through `main.py`:

1. **Parameter Grid Search (`solve.py`):**
   - For all 20 questions, evaluates all 16 configurations in the Cartesian product:
     $$\text{tier} \in \{\text{"cheap"}, \text{"large"}\}, \; \text{style} \in \{\text{"terse"}, \text{"verbose"}\}, \; \text{fewshot} \in \{\text{False}, \text{True}\}, \; \text{context} \in \{\text{False}, \text{True}\}$$
   - Simulates outputs via `mu.build_call(...)` and asserts `mu.question_passes(...)`.
2. **Constrained Cost Minimization:**
   - Filters out failing combinations and picks the minimum-token configuration per query:
     $$c^*(q) = \arg\min_{c \in \mathcal{C}_{\text{pass}}(q)} \text{Tokens}(q, c)$$
   - *Group A (8 queries: `q01`, `q03`, `q09`, `q11`, `q14`, `q15`, `q16`, `q19`):* Routed to `cheap / terse / fewshot=False / context=False`.
   - *Group B (12 queries with `large_only` slots):* Dynamically routes between `cheap + context` and `large` based on snippet token length.
3. **Execution & Packaging (`main.py`):**
   - Exports the chosen table to `routing_config.json`.
   - Runs full batch execution against `fact_bank.json`, writing validated responses to `answers.json` and generating `submission.json`.

---

## 4. Results & Artifacts
- **Status:** PASSED
- **Score:** 200 / 200 points
- **Recovered Flag:** `SIGMOID_TOKENDIET`
- **Resource Metrics:**
  - **Accuracy:** **20 / 20 questions passed (100.0%)**;
  - **Token Consumption:** **1,160 tokens** (vs 1,200 limit and 2,277 baseline, achieving a **49.1% reduction**).
- **Output Artifacts:**
  - `routing_config.json` — optimal per-query dispatch matrix;
  - `answers.json` — model response completions;
  - `submission.json` — official benchmark submission payload.
