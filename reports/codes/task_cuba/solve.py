"""End-to-end reproduction for task_cuba (Zemí).

    /home/kozak/mllm/py solve.py

Rebuilds index/ from the downloaded corpus (about 1-2 min), copies the
reranker checkpoint, and writes task_cuba.zip with exactly main.py + index/.
"""
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SUBMISSION = os.path.join(HERE, "submission")
ZIP = os.path.join(HERE, "task_cuba.zip")

# 1. build index/ (documents, chunk index, int8 vectors, int8 model2vec)
import build_index

build_index.main()

# 2. package
if os.path.exists(ZIP):
    os.remove(ZIP)
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for root, _, files in os.walk(SUBMISSION):
        for f in sorted(files):
            p = os.path.join(root, f)
            z.write(p, os.path.relpath(p, SUBMISSION))
print("wrote", ZIP, round(os.path.getsize(ZIP) / 1e6, 2), "MB")
