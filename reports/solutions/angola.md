# Solution Report: task_angola ("The Waterline Replay")

## 1. Overview & Objective
- **Task ID:** `task_angola` (Angola).
- **Difficulty & Score:** Medium, 200 / 200 points.
- **Track:** LLM Agent Forensics, RAG Memory Audit.
- **Context:** An AI municipal service assistant in Luanda, Angola, hallucinated an erroneous emergency water shutoff notice for the Kilamba district.
- **Input Data:** `test-input.json` containing structured execution trace logs across multiple autonomous LLM agent runs. The `events` array records:
  - Run identifiers (`run_id`) and timestamps;
  - Event types (`event`);
  - Tool invocations and responses;
  - Context retrieval items from long-term memory (`memory.search`);
  - Prompt payloads and generated assistant completions.
- **Objective:** Audit the agent trace log to isolate the corrupt knowledge base entry that triggered the hallucination, extracting its exact `source_id`.
- **Target Flag Format:** String identifier `SIGMOID_<PHRASE>`.

---

## 2. Key Insights & Methodology

RAG pipeline errors stem from query distortion, corrupted memory retrieval, or synthesis failure. In this forensic challenge, the corrupted knowledge item is directly tagged in the trace events of the agent's long-term memory retrieval layer (`memory.search`):
1. **Automated Event Stream Inspection:** Parse `test-input.json` and iterate over `events`.
2. **Filter Retrieval Operations:** Detect events where `ev.get("event") == "memory.search"`.
3. **Identifier Extraction:** Inspect `result_ids` within matching events for strings bearing the platform flag prefix `SIGMOID_`.

---

## 3. Step-by-Step Implementation Details

### Implementation Script (`solve.py`)
The exploit script performs deterministic parsing and platform submission:

```python
import json
from pathlib import Path
from optimizations.sigmoid_api import SigmoidClient


def main():
    path = Path("challenges/task_angola/data/test-input.json")
    data = json.loads(path.read_text(encoding="utf-8"))

    flag = None
    for ev in data["events"]:
        if ev.get("event") == "memory.search" and "SIGMOID_" in str(ev.get("result_ids")):
            for rid in ev.get("result_ids", []):
                if rid.startswith("SIGMOID_"):
                    flag = rid
                    break

    print(f"Identified Faulty Memory ID: {flag}")
    if flag:
        client = SigmoidClient()
        res = client.submit_text("task_angola", flag)
        print("Submission result:", res)
    return flag


if __name__ == "__main__":
    main()
```

---

## 4. Results & Artifacts
- **Status:** `PASSED` (Score: **200 / 200** points).
- **Submission Attempt:** Attempt 1 (`2026-11-14T10:38:00.082Z`, submitter: `Kozak`, 0 incorrect attempts).
- **Extracted Flag:**
  ```text
  SIGMOID_MEM_KILAMBA
  ```
- **Key Artifacts:**
  - `solve.py` (859 bytes, automated trace auditor and submitter).
