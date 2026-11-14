#!/usr/bin/env bash
# Regenerate retrieved_ids.json with the repaired pipeline.
# Usage: ./run_pipeline.sh   (override Python with PYTHON=/path/to/python)
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-/home/kozak/mllm/py}"
export HF_HOME="${HF_HOME:-/tmp/opencode/hf}"
cd work
"$PYTHON" ../retrieval_pipeline.py --output ../retrieved_ids.json
