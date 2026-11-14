#!/usr/bin/env python3
"""Reproduce and verify the "Weak Layer" flag-encoding on the practice table.

Run:  /home/kozak/mllm/py verify_practice.py

The script:
  1. loads the provided 6,000-row practice table,
  2. fits ``DecisionTreeClassifier(random_state=0)`` with default
     hyperparameters,
  3. prints the whole tree and the internal-node question order,
  4. asserts that the first letters of the split features spell DEMO,
  5. runs the submitted ``main.py`` end-to-end in a temp dir and compares
     byte-for-byte with the provided ``train-output.txt``.
"""

from __future__ import annotations

import filecmp
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
TASK = Path("/home/kozak/mllm/MML/work/tasks/task_switzerland")
TRAIN = TASK / "data" / "train-input.csv"
EXPECTED = TASK / "data" / "train-output.txt"


def main() -> int:
    df = pd.read_csv(TRAIN)
    assert df.shape == (6000, 28), df.shape
    X = df.drop(columns=["slab_released"])
    y = df["slab_released"]

    clf = DecisionTreeClassifier(random_state=0)  # library defaults + int seed
    clf.fit(X, y)
    tree = clf.tree_

    print(f"nodes={tree.node_count} leaves={tree.n_leaves} depth={clf.get_depth()} "
          f"train accuracy={clf.score(X, y):.6f}")

    # Render the tree (pre-order, i.e. exactly the order of tree_.feature).
    def show(node: int, indent: str = "") -> None:
        feature = int(tree.feature[node])
        if feature >= 0:
            name = X.columns[feature]
            print(f"{indent}[node {node}] {name} <= {tree.threshold[node]:.6g} "
                  f"(n={tree.n_node_samples[node]})")
            show(int(tree.children_left[node]), indent + "    ")
            show(int(tree.children_right[node]), indent + "    ")
        else:
            print(f"{indent}[leaf {node}] class={tree.value[node].ravel().argmax()} "
                  f"(n={tree.n_node_samples[node]})")

    show(0)

    split_features = [X.columns[int(f)] for f in tree.feature if f >= 0]
    letters = [str(name)[0].upper() for name in split_features]
    word = "".join(letters)
    print("\ninternal-node question order:", " -> ".join(split_features))
    print("initials, in that order       :", " ".join(letters), "=", word)

    expected_flag = EXPECTED.read_text(encoding="utf-8").strip().strip("\r\n")
    assert expected_flag == "SIGMOID_DEMO", expected_flag
    assert f"SIGMOID_{word}" == expected_flag, (word, expected_flag)
    print(f"\nPRACTICE CHECK: SIGMOID_{word} == {expected_flag}  [OK]")

    # End-to-end run of the submitted program.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        shutil.copy(TRAIN, tmp_path / "test-input.csv")
        shutil.copy(HERE / "main.py", tmp_path / "main.py")
        subprocess.run([sys.executable, "main.py"], cwd=tmp_path, check=True)
        produced = (tmp_path / "test-output.txt").read_text(encoding="utf-8")
        print(f"main.py wrote: {produced.strip()!r}")
        assert produced.strip() == expected_flag
    print("END-TO-END CHECK: main.py reproduces the practice flag  [OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
