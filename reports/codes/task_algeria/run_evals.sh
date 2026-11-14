#!/bin/bash
set -x
cd /home/kozak/mllm/MML/work/tasks/task_algeria/data
S=/home/kozak/mllm/MML/work/solutions/task_algeria
for v in "$@"; do
  case "$v" in
    v1) A=$S/modal_out/adapter_r32e4.gguf ;;
    v2) A=$S/modal_out/adapter_r32e4v2.gguf ;;
    *) A=$S/modal_out/$v.gguf ;;
  esac
  /home/kozak/mllm/py evaluation_script.py --model "$A" --data public --out $S/modal_out/res_${v}_public.json > $S/modal_out/res_${v}_public.stdout 2> $S/modal_out/res_${v}_public.stderr
  echo "$v public rc=$?"
  /home/kozak/mllm/py evaluation_script.py --model "$A" --data $S/probe --out $S/modal_out/res_${v}_probe.json > $S/modal_out/res_${v}_probe.stdout 2> $S/modal_out/res_${v}_probe.stderr
  echo "$v probe rc=$?"
done
echo DONE
