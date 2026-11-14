#!/usr/bin/env python3
"""Task: The Weak Layer (task_switzerland).

The flag is not in the tree's predictions -- it is hidden in the fitted tree's
*structure*.  A default ``DecisionTreeClassifier`` is fitted on the snow-pit
table; walking its internal nodes in the order scikit-learn stores them (a
depth-first, pre-order traversal of the tree, i.e. the order in which the
snowpack "asks its questions") and reading the first letter of each split
feature's name spells the hidden word.

Sanity check on the practice table: the split features (in that order) are
``depth_hoar_pct``, ``elevation_m``, ``mean_grain_roundness``,
``overburden_load_kpa`` -> D E M O -> ``SIGMOID_DEMO``, which matches the
provided ``train-output.txt`` exactly.

The program reads ``test-input.csv`` from the current working directory (bare
filename) and writes ``test-output.txt`` into that same directory.
"""

from __future__ import annotations

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

INPUT_FILE = "test-input.csv"
OUTPUT_FILE = "test-output.txt"
FLAG_PREFIX = "SIGMOID_"

# A fixed integer random_state is required by the task statement.
RANDOM_STATE = 0


def load_table(path: str = INPUT_FILE) -> tuple[pd.DataFrame, pd.Series]:
    """Load the snow-pit table and split off the release label."""
    df = pd.read_csv(path)

    # The practice table's label column is ``slab_released``; fall back to the
    # last column for a schema variant, and to a binary column otherwise.
    def is_binary(series: pd.Series) -> bool:
        values = set(series.dropna().unique().tolist())
        return values.issubset({0, 1, 0.0, 1.0, True, False}) and len(values) <= 2

    if "slab_released" in df.columns:
        target = "slab_released"
    elif df.columns[-1].strip().lower() in {
        "label",
        "target",
        "released",
        "slab_released",
        "y",
    }:
        target = df.columns[-1]
    elif is_binary(df[df.columns[-1]]):
        target = df.columns[-1]
    else:
        candidates = [c for c in df.columns if is_binary(df[c])]
        target = candidates[-1] if candidates else df.columns[-1]

    df = df.dropna(subset=[target]).reset_index(drop=True)
    X = df.drop(columns=[target])
    # A stray unnamed row index (e.g. from to_csv) is never a snow-pit
    # question and must not pollute the question sequence.
    X = X.drop(
        columns=[c for c in X.columns if str(c).startswith("Unnamed:")],
        errors="ignore",
    )
    y = df[target]

    # Keep the classifier happy even if a column happens to be non-numeric:
    # category codes preserve the split information without any external data.
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.factorize(X[col], sort=True)[0]

    return X, y


def recover_word(X: pd.DataFrame, y: pd.Series) -> str:
    """Fit the default tree and read the split-feature initials in order."""
    clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
    clf.fit(X, y)
    tree = clf.tree_

    # ``tree_.feature`` is indexed by node id; scikit-learn assigns node ids in
    # depth-first pre-order (root, left subtree, right subtree).  Internal
    # nodes have feature >= 0; -1 marks a leaf.  Reading them in this order
    # yields the sequence of questions the snowpack asks.
    names = [str(c).strip() for c in X.columns]
    letters: list[str] = []
    for node in range(tree.node_count):
        feature = int(tree.feature[node])
        if feature >= 0:
            letters.append(names[feature][0].upper())

    return "".join(letters)


def main() -> None:
    X, y = load_table(INPUT_FILE)
    word = recover_word(X, y)
    flag = f"{FLAG_PREFIX}{word}"
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(flag + "\n")


if __name__ == "__main__":
    main()
