# Solution Report: task_mexico ("Ruta de las Identidades")

## 1. Overview & Objective
- **Task ID:** `task_mexico` (Mexico).
- **Task Type:** Entity Resolution / Record Linkage across municipal registry tables.
- **Submission Mode:** Autonomous script `main.py` evaluated offline.
- **Dataset:** `train-input.csv` (and hidden `test-input.csv`) containing 19,520 records across 3 independent sources:
  - Table `a`: 6,609 records
  - Table `b`: 6,446 records
  - Table `c`: 6,465 records
  - Attributes: `source_table, record_id, first_name, last_name, address`.
- **Ground Truth Structure (`train-output.json`):**
  - Contains 12,038 labeled matching pairs spanning 10,000 unique individuals:
    - 4,484 clusters of size 2 (a-c: 1,549; a-b: 1,507; b-c: 1,428).
    - 2,518 clusters of size 3 (records present in all three tables, yielding $C_3^2 = 3$ pairwise edges each).
    - 2,998 singletons.
  - **Structural Invariant:** At most one record per person per source table. Pairwise linkage is a partial 1-to-1 matching across tables.
- **Output Contract:** `main.py` reads input CSV from CLI args or stdin/cwd, generates `test-output.json`, and outputs `{"matches": [["a_1", "b_2"], ...]}`.

---

## 2. Key Insights & Methodology

1. **Noise Asymmetry (Names vs. Addresses):**
   - **Name Fields:** Severely degraded by OCR errors, typos, swapped first/last names, nicknames, initials, and prefix truncations (e.g., `Mercedes` $\leftrightarrow$ `Lbarra`). Fuzzy name similarity drops to 0.50 among true pairs, making names unsuitable as primary blocking keys.
   - **Address Fields:** Highly conservative; address text is subject only to reformattings and standard Spanish abbreviations, preserving the underlying multiset of semantic tokens:
     - `C.jón` $\leftrightarrow$ `Callejón`, `Int.` $\leftrightarrow$ `Interior`, `Av.` $\leftrightarrow$ `Avenida`, `Peat.` $\leftrightarrow$ `Peatonal`, `Priv.` $\leftrightarrow$ `Privada`, `Depto.` $\leftrightarrow$ `Departamento`, `Edif.` $\leftrightarrow$ `Edificio`, `Prol.` $\leftrightarrow$ `Prolongación`, `Cor.` $\leftrightarrow$ `Corredor`, `Ret.` $\leftrightarrow$ `Retorno`, `Ampl.` $\leftrightarrow$ `Ampliación`, `Calz.` $\leftrightarrow$ `Calzada`, `Circ.` $\leftrightarrow$ `Circunvalación`, `And.` $\leftrightarrow$ `Andador`, `Blvd.` $\leftrightarrow$ `Boulevard`, `Q. ROO` $\leftrightarrow$ `QROO`.

2. **Empirical Address Separability Gap:**
   - Under canonical token normalization, **all 12,038 ground-truth pairs share identical address token multisets** (Jaccard similarity $J = \mathbf{1.000}$).
   - The maximum address Jaccard similarity among all 47,915 non-matching candidate pairs is only **0.417**.
   - No two distinct persons in the same source table share an address ($J \le 0.42$). A threshold of $J \ge 0.60$ guarantees 100% precision and zero false positives.

---

## 3. Implementation Details

Implemented in `main.py` using Python's standard library (`sys`, `csv`, `json`, `re`, `unicodedata`, `collections`, `difflib`) with zero external dependencies:

1. **Canonical Text Normalization:**
   - Strips diacritics via Unicode NFKD decomposition (`unicodedata.normalize('NFKD', ...)`).
   - Converts to lowercase, translates punctuation to spaces, and expands Spanish address abbreviations using compiled regexes (e.g., `c jon` $\to$ `callejon`, `q roo` $\to$ `qroo`).

2. **Multi-Key Blocking Engine:**
   - Candidate pairs are indexed across multiple redundant keys to tolerate isolated noise:
     1. Exact 5-digit postal code (ZIP).
     2. 4-digit ZIP prefix (guards against single-digit errors).
     3. House number + street prefix token.
     4. Exact full name tuple.
     5. Rare name tokens (document frequency $df \le 25$).
   - Yields 83,941 candidate pairs on training data, capturing 100% of true matches.

3. **Scoring & Rescue Branch:**
   - Evaluates address token Jaccard similarity $J$.
   - If $J \ge 0.60$, the pair is unconditionally accepted.
   - For borderline scores ($0.20 \le J < 0.60$), a fallback branch evaluates character-bigram Dice similarity, token fuzzy F1, `SequenceMatcher` ratios, and name agreement ($\ge 0.55$), requiring agreement on core address anchors (ZIP code or house number + street).

4. **Conflict-Guarded Union-Find:**
   - Merges candidate pairs in decreasing order of confidence.
   - Enforces a hard constraint: prohibits merging clusters that contain records from the same `source_table`.
   - Emits all pairwise clique combinations for resulting size-3 components.

5. **Robustness & Validation:**
   - Tested via `stress_test.py` against synthetic corruption of 8% of records (typos, dropped tokens, altered ZIPs), recovering 12,035 / 12,038 true pairs with 0 false positives.
   - Robust CSV reader handles BOM, CRLF, custom separators, and variable column orders.

---

## 4. Results & Artifacts
- **Submission Artifact:** `main.py` (autonomous, standard library only).
- **Training Set Performance (19,520 records):**
  - Recovered Matches: **12,038 / 12,038 (100.0%)**
  - False Positives: **0**
  - False Negatives: **0**
  - **F1 Score: 1.0000**
- **Test Set Evaluation & Flag:**
  - Evaluated on platform test split: **12,077 / 12,077 matches (100/100 points, PASSED)**.
  - **Flag:** **`SIGMOID_RECORD_LINKAGE`**
- **Runtime:** $1.5 \dots 5.0$ seconds total runtime on a single CPU core.
