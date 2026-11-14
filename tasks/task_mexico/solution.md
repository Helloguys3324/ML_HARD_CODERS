# task_mexico — "Ruta de las Identidades"

**Deliverable:** `work/solutions/task_mexico/main.py` (self-contained, standard
library only). Reads the hidden `test-input.csv`, writes `test-output.json` and
prints the same `{"matches": [...]}` JSON on stdout.

**Confidence: HIGH** — the pipeline reproduces the labelled practice output
exactly (12,038/12,038 pairs, 0 missing, 0 extra) and the score threshold sits
in an enormous empirical gap.

---

## 1. Data / EDA (labeled practice set)

`data/train-input.csv` — 19,520 records from three source tables
(`a`: 6,609, `b`: 6,446, `c`: 6,465); columns
`source_table, record_id, first_name, last_name, address`.
`data/train-output.json` — 12,038 labeled pairs.

Union-find over the labeled pairs shows the ground-truth structure:

| cluster size | clusters | sources present |
|---|---|---|
| 2 | 4,484 | two different tables (a-c 1,549 / a-b 1,507 / b-c 1,428) |
| 3 | 2,518 | all three tables |
| 1 | 2,998 singletons | one table |

→ 10,000 distinct persons; **at most one record per person per source table**,
so pairwise matching is a partial 1-to-1 matching between each pair of tables.

### The decisive signal

Addresses are randomized but the register only *reformats* them per record:
abbreviations (`C.jón` ↔ `Callejón`, `Int.` ↔ `Interior`, `Av.` ↔ `Avenida`,
`Peat.` ↔ `Peatonal`, `Priv.` ↔ `Privada`, `Depto.` ↔ `Departamento`,
`Edif.` ↔ `Edificio`, `Prol.` ↔ `Prolongación`, `Cor.` ↔ `Corredor`,
`Ret.` ↔ `Retorno`, `Ampl.` ↔ `Ampliación`, `Calz.` ↔ `Calzada`,
`Circ.` ↔ `Circunvalación`, `And.` ↔ `Andador`, `Blvd.` ↔ `Boulevard`,
`Q. ROO` ↔ `QROO`), accents/punctuation, token order, and (for names only)
typos/swaps/truncations.

After canonicalisation:

* **all 12,038 true pairs have the identical canonical address token
  *multiset*** (address token-set Jaccard = **1.000**);
* the maximum address Jaccard over all **47,915 non-matching candidate pairs**
  is **0.417**;
* no two records from the *same* source table ever share a near-identical
  address (max Jaccard 0.42), i.e. an address identifies a person.

Names are much noisier (typos, first/last swaps, initials, left-truncation such
as `Mercedes` ↔ `Lbarra`, min fuzzy name score 0.50 among true pairs), so they
are used only as a secondary confirmation.

## 2. Pipeline in `main.py`

1. **Normalisation** — NFKD accent stripping, lowercase, punctuation → space;
   address abbreviation expansion (incl. multi-token merges `c jon` →
   `callejon`, `q roo` → `qroo`); name tokens from first+last.
2. **Blocking** (several independent keys, so no single corrupted field can
   hide a true pair): zip5, zip4 (one-digit zip damage), (house number,
   street-name prefix), exact full-name tuple, and rare name tokens (df ≤ 25).
   83,941 candidate pairs on the practice data; all 12,038 labels are covered.
3. **Scoring** — canonical token-set Jaccard `J`; for pairs in the borderline
   band (`0.20 ≤ J < 0.60`) also fuzzy token F1 (prefix/initial aware),
   character-bigram Dice and `difflib` ratio on the canonical address string;
   fuzzy name F1 as a secondary check.
4. **Decision** — `J ≥ 0.60` accepts outright; rescue branch requires
   `combined ≥ 0.80`, `name ≥ 0.55` **and** a stable address anchor (equal
   zip5, or equal house number + matching street name, or ≥ 2 shared numeric
   components). Same-table pairs require `J ≥ 0.90` and `name ≥ 0.60`
   (never fires on the practice data).
5. **Clustering** — union-find over accepted edges (highest score first) with
   a hard "one record per source table" conflict guard; every pair inside a
   component is emitted (triples → 3 pairs), exactly as the spec requires.

Ties/order: ids sorted inside each pair, pairs sorted; order is irrelevant to
the grader.

## 3. Validation (reproducible)

```bash
cd /home/kozak/mllm/MML/work/solutions/task_mexico
/home/kozak/mllm/py validate.py      # copies train-input.csv as test-input.csv, runs main.py
```

```
[train] rc=0 pairs=12038 stderr=[task_mexico] 12038 pairs from .../test-input.csv
[train] expected=12038 got=12038 missing=0 extra=0
PASS
```

* end-to-end runtime on 19,520 records: **≈5–8 s** (pure stdlib, single core);
  a clean scratch-directory run (`cp main.py` + `test-input.csv`) produced
  `test-output.json` in 7.4 s with identical bytes on stdout;
* `stress_test.py` applies far harsher synthetic damage than the register
  (8% per-record chance each of: zip digit change, dropped address token,
  unseen abbreviation, street-name replacement, name typo, name swap, initial
  truncation): 12,035/12,038 recovered, **0 extra pairs**. The 3 misses are
  cases where a street name *and* the zip were both replaced — damage the
  actual generator never applies;
* I/O robustness tested: header-only file (→ empty matches), CRLF, BOM,
  reordered columns, aliased headers (`id`/`full_address`/`name`), `;`
  delimiter, duplicate ids, input path via `argv[1]`, stdout + file output.

## 4. Grader I/O convention

Other LMML file tasks (e.g. task_egypt/task_iceland) run `python main.py` in a
scratch directory containing `test-input.<ext>` and read
`test-output.<ext>`. `main.py` therefore:

* locates the input via `argv[1]`, env vars, `test-input.csv`,
  `train-input.csv`, common names, or any CSV with a matching header;
* writes `test-output.json` in the working directory (and next to the input),
  and prints the same JSON object to stdout;
* exposes `solve(csv_path) -> {"matches": [...]}` for import-based harnesses.

## 5. Reproduce the evidence

`evidence.txt` — counts and separation margins (generated by the run above).
`validate.py`, `stress_test.py` — exact-match validation and corruption stress
test. Raw data: `work/tasks/task_mexico/data/`.
