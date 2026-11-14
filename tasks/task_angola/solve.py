import json
from pathlib import Path
from optimizations.sigmoid_api import SigmoidClient


def main():
    path = Path("challenges/task_angola/data/test-input.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    
    # Locate the memory search event that caused the incident
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
