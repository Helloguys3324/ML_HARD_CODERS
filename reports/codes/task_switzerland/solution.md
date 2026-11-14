# task_switzerland — "The Weak Layer" (Easy, file submission, 100 pts)

**Deliverable:** `main.py` — reads `test-input.csv` from the current working
directory (bare filename), trains a default `DecisionTreeClassifier`, derives
the hidden word from the *structure* of the fitted tree, and writes
`SIGMOID_<WORD>` to `test-output.txt` in the same directory.

**Status:** implemented, verified against the provided practice pair
(`train-input.csv` → `SIGMOID_DEMO`, byte-identical), verified under the exact
pinned stack `scikit-learn==1.7.2 / pandas==2.3.3 / numpy==2.2.6`, stress-tested
on subsamples/shuffles and schema variants. **Not submitted** (the orchestrator
submits).

**Confidence: high** — the practice table's hidden answer is reproduced exactly
by the derived rule, and the rule is the only natural reading of the tree that
produces a word.

---

## 1. The trick (what the task actually asks)

The statement says explicitly: the answer is *not* in the predictions ("a
well-fitted tree here will be close to perfect, and that tells you nothing"); it
is encoded in the **structure of the fitted tree**, and you should fit with the
**library's default hyperparameters** and an **integer `random_state`**.

The practice tree is perfectly pure (train accuracy 1.000, depth 4, 9 nodes,
5 leaves), so predictions carry no information. Its internal-node structure is:

```
[node 0] depth_hoar_pct        <= 20    (n=6000)
    [leaf 1] class=0 (n=1400)
    [node 2] elevation_m       <= 1850  (n=4600)
        [leaf 3] class=0 (n=1100)
        [node 4] mean_grain_roundness <= 0.55 (n=3500)
            [node 5] overburden_load_kpa <= 1.5 (n=2650)
                [leaf 6] class=0 (n=650)
                [leaf 7] class=1 (n=2000)
            [leaf 8] class=0 (n=850)
```

Read the split features in the order the tree stores its nodes
(`tree_.feature`, i.e. depth-first **pre-order** — the order in which the
snowpack asks its questions):

```
depth_hoar_pct -> elevation_m -> mean_grain_roundness -> overburden_load_kpa
      D               E                    M                    O
```

**D E M O → `SIGMOID_DEMO`**, exactly the provided `train-output.txt`.

## 2. Why this is the intended encoding (not a coincidence)

The same tree yields nonsense under every other plausible ordering:

| Ordering | Letter sequence |
|---|---|
| **pre-order node order (`tree_.feature`)** | **D E M O  → DEMO ✅** |
| in-order traversal | D E O M |
| post-order traversal | O M E D |
| reverse node order | O M E D |
| feature importance descending | O M E D |
| feature index ascending | O M D E |
| split threshold ascending | M O D E |
| split threshold descending | E D O M |

Only the pre-order split-question sequence is a word, and it matches the
published practice answer. scikit-learn assigns node ids in depth-first
pre-order, which is also the top-to-bottom order shown by `export_text` /
`plot_tree`, so "read its structure" = read `tree_.feature` in node-index
order.

The tree is a planted conjunctive rule (`depth_hoar_pct > 20 AND
elevation_m > 1850 AND mean_grain_roundness > 0.55 AND
overburden_load_kpa > 1.5 → released`), one condition per letter, which is why
the question order spells the word. The hidden `test-input.csv` is generated
the same way from a different word; the program derives it rather than
hard-coding anything.

## 3. Implementation (`main.py`)

1. `pd.read_csv("test-input.csv")` (bare filename, cwd).
2. Target column: `slab_released` if present; otherwise the last column if it
   is binary; otherwise the last binary column. A stray `Unnamed: 0` index
   column is dropped; any non-numeric column is factorized (defensive only —
   the practice schema is all numeric).
3. `DecisionTreeClassifier(random_state=0)` — library defaults, integer seed,
   no depth/leaf/feature constraints (as required: constraining would change
   the shape and lose the word).
4. Iterate node ids `0..node_count-1`, keep `tree_.feature[node] >= 0`
   (internal nodes only) and append `str(column_name)[0].upper()`.
5. Write `SIGMOID_<word>\n` to `test-output.txt`.

No external data, no pretrained models, no per-task constants.

## 4. Local evidence

Environment (exact pinned versions from the statement, installed in a throwaway
venv `/tmp/opencode/venv172`):

```
python 3.13.17
sklearn 1.7.2
pandas 2.3.3
numpy 2.2.6
```

### 4.1 Practice verification — `verify_practice.py`

```
nodes=9 leaves=5 depth=4 train accuracy=1.000000
internal-node question order:
  depth_hoar_pct -> elevation_m -> mean_grain_roundness -> overburden_load_kpa
initials, in that order: D E M O = DEMO
PRACTICE CHECK: SIGMOID_DEMO == SIGMOID_DEMO  [OK]
main.py wrote: 'SIGMOID_DEMO'
END-TO-END CHECK: main.py reproduces the practice flag  [OK]
```

Full log: `evidence/verify_practice_sklearn_1_7_2.txt`.

### 4.2 Stability

* `random_state` 0–100 all produce the same tree and the same word (`DEMO`).
* 10 random subsamples at each of 50/60/70/80/90/95 % of rows → word is
  `DEMO` in every case.
* Shuffled row order → `DEMO`.
* End-to-end `main.py` on a shuffled 90 % subsample → `SIGMOID_DEMO`.
* Schema variants all still output `SIGMOID_DEMO`: renamed target
  (`released`, `y`), stray `Unnamed: 0` index column, an extra categorical
  string feature column.

Logs: `evidence/stress_subsample_shuffle.txt`,
`evidence/library_versions.txt`.

### 4.3 Local run commands

```bash
# practice run, from a scratch dir containing test-input.csv + main.py
cp work/tasks/task_switzerland/data/train-input.csv /tmp/ch/test-input.csv
cp work/solutions/task_switzerland/main.py /tmp/ch/
cd /tmp/ch && /home/kozak/mllm/py main.py
cat test-output.txt          # SIGMOID_DEMO

# reproducible verification (also works with the pinned 1.7.2 venv)
cd work/solutions/task_switzerland && /home/kozak/mllm/py verify_practice.py
```

## 5. Deliverable checklist

- [x] Reads `test-input.csv` by bare filename from cwd.
- [x] Writes `test-output.txt` in the same directory, one line, uppercase,
      no quotes/text, trailing newline allowed.
- [x] Derives the word from the fitted tree structure (feature-name initials
      in pre-order split order), not from predictions and not hard-coded.
- [x] Default `DecisionTreeClassifier` hyperparameters + integer `random_state`.
- [x] No external data / APIs / pretrained models.

## 6. Remaining risk

Low. The only ambiguity is the traversal order if the hidden fitted tree were
to branch across levels (pre-order vs breadth-first); the practice tree has
exactly one internal node per depth (a planted conjunctive chain), and hidden
tables generated the same way will too. Pre-order (`tree_.feature` order) is
scikit-learn's canonical storage/`export_text` order and is the reading that
reproduces the published practice word, so it is used.

`answer.txt` contains the practice flag `SIGMOID_DEMO` (verification only —
the graded artifact is `main.py`, which derives the real hidden word from
`test-input.csv` at evaluation time).
