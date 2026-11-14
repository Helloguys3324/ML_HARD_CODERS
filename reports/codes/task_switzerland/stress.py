import pandas as pd, numpy as np, subprocess, sys, tempfile, os, shutil
from sklearn.tree import DecisionTreeClassifier

rng = np.random.default_rng(0)
df = pd.read_csv("/home/kozak/mllm/MML/work/tasks/task_switzerland/data/train-input.csv")
X = df.drop(columns=["slab_released"]); y = df["slab_released"]

def word_of(Xf, yf, rs=0):
    t = DecisionTreeClassifier(random_state=rs).fit(Xf, yf).tree_
    return "".join(str(Xf.columns[f])[0].upper() for f in t.feature if f >= 0)

print("baseline:", word_of(X, y))
for frac in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
    words = set()
    for seed in range(10):
        idx = rng.choice(len(X), size=int(frac*len(X)), replace=False)
        words.add(word_of(X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)))
    print(f"subsample {frac:.2f} -> {words}")

# shuffled rows
perm = rng.permutation(len(X))
print("shuffled rows ->", word_of(X.iloc[perm].reset_index(drop=True), y.iloc[perm].reset_index(drop=True)))

# end-to-end: main.py on shuffled/sub sampled table
d = tempfile.mkdtemp(dir="/tmp/opencode")
idx = rng.choice(len(X), size=5400, replace=False)
test = df.iloc[idx].sample(frac=1.0, random_state=1).reset_index(drop=True)
test.to_csv(os.path.join(d, "test-input.csv"), index=False)
shutil.copy("main.py", d)
subprocess.run([sys.executable, "main.py"], cwd=d, check=True)
print("e2e output:", open(os.path.join(d, "test-output.txt")).read().strip())
shutil.rmtree(d)
