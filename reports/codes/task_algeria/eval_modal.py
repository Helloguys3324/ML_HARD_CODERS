"""Run the shipped evaluation_script.py on Modal (CPU container with llama-cpp-python).

Usage:
    /home/kozak/mllm/py -m modal run eval_modal.py --adapter task_algeria/adapter_r32e4.gguf

The base Q8_0 GGUF must already be in the lmml-data volume at
task_algeria/models/smollm2-360m-instruct-q8_0.gguf.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
DATA = Path("/home/kozak/mllm/MML/work/tasks/task_algeria/data")
APP = "task-algeria-eval"

data_vol = modal.Volume.from_name("lmml-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "cmake", "ninja-build")
    .uv_pip_install("llama-cpp-python==0.3.36", "gguf==0.19.0", "numpy", "safetensors", "huggingface_hub")
    .env({"TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(str(DATA / "evaluation_script.py"), "/root/task/evaluation_script.py")
    .add_local_file(str(DATA / "style_checker.py"), "/root/task/style_checker.py")
    .add_local_file(str(DATA / "register" / "RULES.md"), "/root/task/register/RULES.md")
    .add_local_file(str(DATA / "register" / "vocabulary.json"), "/root/task/register/vocabulary.json")
    .add_local_file(str(DATA / "public" / "prompts.json"), "/root/task/public/prompts.json")
    .add_local_file(str(DATA / "public" / "calls.json"), "/root/task/public/calls.json")
)

app = modal.App(APP, image=image)


@app.function(timeout=3600, volumes={"/data": data_vol})
def evaluate(adapter: str | None = None, tag: str = "eval", data_subdir: str = "public") -> dict:
    import os
    import subprocess
    import sys

    task = Path("/root/task")
    models = task / "models"
    models.mkdir(exist_ok=True)
    src = Path("/data/task_algeria/models/smollm2-360m-instruct-q8_0.gguf")
    dst = models / src.name
    if src.is_file() and not dst.exists():
        os.symlink(src, dst)
    out = Path("/data/task_algeria") / f"result_{tag}.json"
    cmd = [sys.executable, "evaluation_script.py", "--data", data_subdir, "--out", str(out)]
    if adapter:
        cmd += ["--model", adapter if os.path.isabs(adapter) else f"/data/{adapter}"]
    print("CMD", cmd, flush=True)
    r = subprocess.run(cmd, cwd=str(task), capture_output=True, text=True)
    print("STDOUT:\n", r.stdout[-20000:], flush=True)
    print("STDERR:\n", r.stderr[-20000:], flush=True)
    try:
        result = json.loads(r.stdout)
    except Exception:
        result = {"parse_error": r.stdout[-2000:], "returncode": r.returncode}
    result["returncode"] = r.returncode
    # persist the reply file for inspection
    if out.is_file():
        result["out_file"] = str(out)
    data_vol.commit()
    return result


@app.local_entrypoint()
def main(adapter: str | None = None, tag: str = "eval", data: str = "public"):
    res = evaluate.remote(adapter=adapter, tag=tag, data_subdir=data)
    print(json.dumps(res, indent=2))
