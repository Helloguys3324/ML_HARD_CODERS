# Solution Report: task_rwanda ("The Drying Floor at Karongi")

## 1. Overview & Objective
- **Task ID:** `task_rwanda` (Rwanda).
- **Difficulty & Score:** Easy, 100 / 100 points.
- **Domain:** General AI / Agent State Machines (LangGraph).
- **Output Format:** Bug fix in `graph.py` yielding flag `SIGMOID_<NAME>`.
- **Premise:**
  - An agricultural coffee cooperative in Karongi (Lake Kivu) automated its drying floor tracking via a LangGraph state machine:
    $$\text{receive\_batch} \longrightarrow \text{inspect\_batch} \longrightarrow (\text{ready?}) \longrightarrow \begin{cases} \text{dry\_batch} \to \text{inspect\_batch} \to \dots & (\text{unapproved}) \\ \text{grade\_batch} \to \text{END} & (\text{approved}) \end{cases}$$
  - In the initial codebase, the workflow was trapped in an infinite loop. Incoming coffee lots, even those within regulatory moisture limits, looped indefinitely between inspection and drying passes until hitting maximum recursion depth (`GraphRecursionError`).
  - The objective was to diagnose and patch the bug in the state machine definition.

---

## 2. Key Insights & Root Cause Analysis
- **Graph State Mechanism:**
  - The workflow executes over a `StateGraph(BatchState)` where state attributes are managed via a `TypedDict`.
  - Conditional routing after measurement is governed by:
    ```python
    def route_after_inspection(state: BatchState) -> Literal["dry_batch", "grade_batch"]:
        if state["approved"]:
            return "grade_batch"
        return "dry_batch"
    ```
- **Bug Localization:**
  - Physical moisture reduction logic in `dry_batch` and final grading in `grade_batch` were implemented correctly.
  - The flaw resided in `inspect_batch`: the node evaluated the boolean threshold `ready = MOISTURE_MIN <= state["moisture"] <= MOISTURE_MAX`, but omitted the `"approved": ready` assignment from the returned state dictionary.
  - Consequently, `state["approved"]` remained frozen at its default value `False` across all execution cycles.

---

## 3. Implementation Details
The fix is contained within `graph.py`:

1. **Domain Parameters:**
   - Permissible dried coffee moisture window: `MOISTURE_MIN = 10.5%`, `MOISTURE_MAX = 12.5%`.
   - Table dry-down increment per cycle: `DRY_STEP = 0.8%`.
   - Grade "A" centering band: `GRADE_TOLERANCE = 0.5%` around 11.5%.
2. **State Definition (`BatchState`):**
   ```python
   class BatchState(TypedDict):
       batch_id: str
       moisture: float
       passes: int
       inspections: int
       received: bool
       approved: bool
       grade: str
   ```
3. **Patching `inspect_batch`:**
   - Modified the return dictionary to explicitly propagate the incremented inspection counter and the updated approval boolean:
     ```python
     def inspect_batch(state: BatchState) -> dict:
         ready = MOISTURE_MIN <= state["moisture"] <= MOISTURE_MAX
         return {"inspections": state["inspections"] + 1, "approved": ready}
     ```
4. **Execution Verification:**
   - Verified that batches with initial compliant moisture (e.g., 12.0%) route directly to `grade_batch` and terminate at `END` in a single pass without recurrence.
   - Out-of-spec batches iterate through `dry_batch` until crossing into $[10.5\%, 12.5\%]$, terminating reliably.

---

## 4. Results & Artifacts
- **Status:** PASSED
- **Score:** 100 / 100 points
- **Recovered Flag:** `SIGMOID_CONVERGENCE`
- **Modified Source File:** `graph.py`
