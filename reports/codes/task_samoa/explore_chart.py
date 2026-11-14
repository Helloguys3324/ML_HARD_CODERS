"""Decode the chart embedded in the engine binary: geometry first, then cipher."""
import json
import math
import re
from pathlib import Path

BIN = Path(__file__).resolve().parents[2] / "tasks/task_samoa/bundle/extracted/island-duel-participant/samoa-linux-x86_64"
line = [l for l in BIN.read_bytes().split(b"\n") if b'"prefix":"SIGMOID_"' in l]
# use strings-like extraction from the whole file instead
raw = BIN.read_bytes()
start = raw.index(b'{"prefix":"SIGMOID_"')
blob = raw[start:start + 4000].decode("utf-8", errors="ignore")
chart, consumed = json.JSONDecoder().raw_decode(blob)
print("chart bytes:", consumed)
print(json.dumps(chart, indent=1))

stars = chart["stars"]
guides = [s for s in stars if s["size"] > 6]
companions = [s for s in stars if s["size"] <= 6]
print("\nGUIDES:", [(s["text"], s["x"], s["y"], s["size"]) for s in guides])
print("COMPANIONS:", len(companions))

for g in guides:
    ds = sorted(
        ((math.hypot(c["x"] - g["x"], c["y"] - g["y"]), c["text"], c["x"], c["y"]) for c in companions)
    )
    print(f"\nGuide {g['text']} at ({g['x']},{g['y']}):")
    for d, t, x, y in ds:
        print(f"   {t}  d={d:8.4f}  ({x},{y})")
