import os
import sys
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def main():
    # 1. Read test-input.csv from its own working directory by bare filename,
    # with fallback to train-input.csv for local testing if test-input.csv is not present.
    input_file = "test-input.csv"
    if not os.path.exists(input_file):
        if os.path.exists("train-input.csv"):
            input_file = "train-input.csv"
        elif os.path.exists("data/train-input.csv"):
            input_file = "data/train-input.csv"
        elif os.path.exists("data/test-input.csv"):
            input_file = "data/test-input.csv"

    df = pd.read_csv(input_file)

    target_col = None
    for col in ["slab_released", "released", "target", "label", "y"]:
        if col in df.columns:
            target_col = col
            break
    if target_col is None:
        target_col = df.columns[-1]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    feature_names = list(X.columns)

    # 2. Fit DecisionTreeClassifier with default parameters and an integer random_state
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)

    tree = clf.tree_
    classes = list(clf.classes_)
    class_1_idx = classes.index(1) if 1 in classes else len(classes) - 1

    # 3. Find the leaf corresponding to the released slopes (class 1)
    best_leaf = None
    max_c1 = -1
    for node_id in range(tree.node_count):
        if tree.children_left[node_id] == -1 and tree.children_right[node_id] == -1:
            c1_count = tree.value[node_id][0][class_1_idx]
            if c1_count > max_c1:
                max_c1 = c1_count
                best_leaf = node_id

    # 4. Trace the decision path from root (0) to best_leaf
    def find_path(current, target):
        if current == target:
            return []
        if tree.children_left[current] == -1 and tree.children_right[current] == -1:
            return None

        left = tree.children_left[current]
        if left != -1:
            p = find_path(left, target)
            if p is not None:
                feat_name = feature_names[tree.feature[current]]
                return [feat_name[0].upper()] + p

        right = tree.children_right[current]
        if right != -1:
            p = find_path(right, target)
            if p is not None:
                feat_name = feature_names[tree.feature[current]]
                return [feat_name[0].upper()] + p

        return None

    word_letters = find_path(0, best_leaf)
    if not word_letters:
        # Fallback: internal nodes in preorder traversal
        word_letters = []
        for i in range(tree.node_count):
            if tree.children_left[i] != -1:
                feat_name = feature_names[tree.feature[i]]
                word_letters.append(feat_name[0].upper())

    word = "".join(word_letters)
    flag = f"SIGMOID_{word}"

    # 5. Write test-output.txt into the working directory
    with open("test-output.txt", "w", encoding="utf-8") as f:
        f.write(f"{flag}\n")

    print(f"Recovered flag: {flag}")


if __name__ == "__main__":
    main()
