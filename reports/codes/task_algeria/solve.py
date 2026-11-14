"""
End-to-end for task_algeria: convert a PEFT LoRA adapter to the GGUF adapter the grader
loads, then score it on the public data with the shipped evaluation_script.

Usage:
    python solve.py --adapter kaggle/out/lora_adapter --out adapter.gguf
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

TASK = Path(__file__).resolve().parent
DATA = TASK / "data"
sys.path.insert(0, str(DATA))


def find_adapter(dirpath: Path) -> Path:
    if (dirpath / "adapter_model.safetensors").is_file():
        return dirpath
    # Kaggle output nests ethernet the folder once
    for cand in dirpath.rglob("adapter_model.safetensors"):
        return cand.parent
    raise FileNotFoundError(f"no adapter_model.safetensors under {dirpath}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", default=str(TASK / "kaggle" / "out" / "lora_adapter"))
    ap.add_argument("--out", default=str(TASK / "adapter.gguf"))
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--outtype", default="f16", choices=["f16", "f32"])
    args = ap.parse_args()

    adapter_dir = find_adapter(Path(args.adapter))
    print(f"[adapter] {adapter_dir}")

    # 1) convert PEFT -> GGUF adapter
    conv = subprocess.run(
        [sys.executable, str(DATA / "tools" / "lora_to_gguf.py"), str(adapter_dir),
         "--outfile", args.out, "--outtype", args.outtype],
        capture_output=True, text=True)
    print(conv.stdout.strip() or "", conv.stderr.strip() or "")

    # 2) evaluate on the public data
    if args.skip_eval:
        return 0
    ev = subprocess.run(
        [sys.executable, str(DATA / "evaluation_script.py"),
         "--model", args.out, "--data", str(DATA / "public"),
         "--out", str(TASK / "replies_final.json")],
        capture_output=True, text=True)
    sys.stderr.write(ev.stderr)
    result = json.loads(ev.stdout)
    print(json.dumps(result, indent=2))
    passed = result.get("passed", False)
    print("\nRESULT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())