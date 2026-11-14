# Solution Report: task_switzerland ("The Weak Layer")

## 1. Overview & Objective
- **Task ID:** `task_switzerland` (Switzerland).
- **Difficulty:** Easy (100 points).
- **Submission Type:** Executable script `main.py` evaluated offline in an automated grading container.
- **Contract & Requirements:**
  - `main.py` reads `test-input.csv` from the current working directory.
  - Trains a scikit-learn `DecisionTreeClassifier` with default hyperparameters and an integer `random_state`.
  - Extracts the secret word from the **internal structure of the trained decision tree** (model predictions are uninformative).
  - Writes the flag `SIGMOID_<WORD>` into `test-output.txt` in the current working directory (single uppercase line, no quotes).
- **Domain Premise:**
  - Avalanche release simulation ("The Weak Layer").
  - The decision tree achieves 1.000 accuracy trivially; the hidden payload is encoded exclusively in the sequence of feature splits.
- **Goal:** Build an autonomous, zero-dependency `main.py` that reconstructs the decision tree, traces the path to the positive avalanche class, extracts the initial letters of splitting features, and writes the flag.

---

## 2. Key Insights & Methodology

### 2.1 Training Tree Topology (`train-input.csv`)
- On `train-input.csv`, the target variable `slab_released` is perfectly partitioned by a tree of depth 4, 9 total nodes, and 5 leaves (accuracy: 1.000):
  - **Node 0:** `depth_hoar_pct <= 20` ($n=6000$) $\to$ Left: Leaf 1 (class 0, $n=1400$); Right: Node 2 ($n=4600$).
  - **Node 2:** `elevation_m <= 1850` ($n=4600$) $\to$ Left: Leaf 3 (class 0, $n=1100$); Right: Node 4 ($n=3500$).
  - **Node 4:** `mean_grain_roundness <= 0.55` ($n=3500$) $\to$ Left: Node 5 ($n=2650$); Right: Leaf 8 (class 0, $n=850$).
  - **Node 5:** `overburden_load_kpa <= 1.5` ($n=2650$) $\to$ Left: Leaf 6 (class 0, $n=650$); Right: Leaf 7 (class 1, avalanche released, $n=2000$).

### 2.2 Decoding Rule
- The sequence of split features on the path from the root node to the positive avalanche leaf (class 1):
  1. **`depth_hoar_pct`** $\to$ **D**
  2. **`elevation_m`** $\to$ **E**
  3. **`mean_grain_roundness`** $\to$ **M**
  4. **`overburden_load_kpa`** $\to$ **O**
- Combining the first letter of each feature name along the decision path yields:
  $$\text{D} \to \text{E} \to \text{M} \to \text{O} \quad \Longrightarrow \quad \mathbf{DEMO} \quad \Longrightarrow \quad \mathbf{SIGMOID\_DEMO}$$
  This perfectly matches the baseline `train-output.txt`.
- Alternative traversals (in-order, post-order, sorted by feature importance or split threshold) produce meaningless anagrams (`DOM`, `OMED`, `MODE`, `EDOM`). Only the root-to-target path produces coherent English words.

---

## 3. Implementation Details

1. **`main.py` (Grading Submission Artifact):**
   - **Resilient Data Ingestion:** Looks for `test-input.csv` in the current directory, with automatic fallback to `train-input.csv` for local testing.
   - **Target Column Identification:** Detects target via candidates `["slab_released", "released", "target", "label", "y"]` or defaults to the last column.
   - **Model Fitting:** Fits `DecisionTreeClassifier(random_state=42)` without artificial depth or leaf limits to preserve exact split topologies.
   - **Target Leaf Localization:** Scans leaves (`children_left[i] == -1` and `children_right[i] == -1`) to find the leaf containing the maximum count of positive class instances (`class 1`).
   - **Path Tracing:** Recursively traces internal nodes from root (0) to the target leaf, extracting the uppercase first letter of each split feature:
     ```python
     feat_name = feature_names[tree.feature[current]]
     return [feat_name[0].upper()] + p
     ```
   - **Fallback Mechanism:** If path tracing fails, falls back to pre-order traversal across all internal nodes (`children_left[i] != -1`).
   - **Output Emission:** Writes `SIGMOID_<WORD>\n` to `test-output.txt`.

2. **`verify_practice.py`:**
   - Fits the tree on `train-input.csv`, renders ASCII tree representation, and asserts exact equality with `train-output.txt` (`SIGMOID_DEMO`).

3. **`stress.py` & Robustness Evidence (`evidence/`):**
   - `stress_subsample_shuffle.txt`: Confirms tree stability across `random_state` values $0 \dots 100$, random row subsampling (50% to 95%), and arbitrary row shuffling—all yielding the identical tree and `DEMO` word.
   - `library_versions.txt`: Validates execution under Python 3.13.17, scikit-learn 1.7.2, pandas 2.3.3, and numpy 2.2.6.

---

## 4. Results & Artifacts
- **Submission Artifact:** `main.py` (autonomous, standard scikit-learn stack).
- **Validation Flag (Train Split):** `SIGMOID_DEMO`
- **Key Artifacts (`task_switzerland/`):**
  - `main.py`: Final submission script compliant with platform constraints.
  - `solution.md`: Technical breakdown of tree topology and traversal logic.
  - `verify_practice.py`: Automated sanity test on practice data.
  - `stress.py`: Subsampling and shuffle stability test harness.
  - `test-output.txt`: Generated local output (`SIGMOID_DEMO`).
  - `answer.txt`: Verification marker (`SIGMOID_DEMO`).
  - `evidence/`: Logs confirming split invariance across random seeds.
