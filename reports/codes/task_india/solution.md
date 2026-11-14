# task_india — Twins Separated at Birth (Easy, file → `main.py`)

## Task
Given a shuffled collection of Indian-monument photos, output **every unordered
pair of images that originate from the same source photograph**. Metric:
pair-level precision / recall / F1; **F1 ≥ 0.88 passes** (100 points).

Data: `work/tasks/task_india/data/train/` — 4,057 JPEGs (`img_000001.jpg` …
`img_004057.jpg`); labels `train-output.csv` — **1,399 duplicate pairs** over
1,119 groups (979 groups of 2, 140 groups of 3; 2,378 images involved, the other
1,679 images are singletons). The hidden evaluation archive is produced by the
same generator with a different shuffle/seed.

## What the duplicates look like (measured, not guessed)
SIFT + RANSAC similarity fits on labelled pairs show the generator:
1. takes a source photograph,
2. crops a **central sub-window** (scale ≈ 0.66–1.1, crop centre within ±7 % of
   the image centre),
3. rescales to its own resolution and re-encodes as JPEG (sometimes with blur).

Consequences used by the solver:
* **Aspect ratio is preserved**: for every labelled pair
  `min(ar_a,ar_b)/max(ar_a,ar_b) ≥ 0.989` (measured; 0 violations in 1,399).
* A duplicate pair is *not* pixel-identical: full-image pHash alone only
  recovers ≈ 55 % of true pairs at hamming ≤ 8 and generates many collisions.
* The archive contains deliberate **hard negatives**: different photographs of
  the same monument (often taken moments apart) that can reach NCC ≈ 0.99 and
  200 RANSAC inliers after alignment. They cannot be avoided with a global
  similarity score alone — hence a two-stage design and a threshold chosen to
  keep precision high without sacrificing recall.

## Method (`main.py`)
1. **Input discovery** — recursively scans the working directory (or the
   directory / zip passed as `argv[1]`); zips are auto-extracted. Output names
   are image basenames.
2. **Multi-zoom pHash** — 64-bit DCT pHash (8×8 low-frequency signs) computed on
   8 centred crops (fractions 1.0, 0.9, 0.84, 0.78, 0.72, 0.66, 0.60, 0.54).
   Candidate distance:
   `d(A,B) = min_j hamming(pHash(A_full), pHash(B_zoom_j))` and symmetrically,
   taking the minimum (a duplicate’s full view matches a zoomed view of the
   source).
3. **Candidate generation** — pair is a candidate if `d ≤ 20` and
   aspect-ratio ratio `≥ 0.97`. On the training archive: 81,622 candidates
   containing ≈ 97 % of all 1,399 true pairs. Computed block-wise with
   vectorised popcount over 4 processes (~8 s for 4k images).
4. **Geometric verification** — ORB (350 keypoints, images ≤ 512 px, Lowe ratio
   0.75) + `estimateAffinePartial2D` RANSAC (3 px), then A is warped onto B and
   the normalised cross-correlation of the overlap is measured. Accept when
   * inliers ≥ 40, inlier fraction ≥ 0.60, NCC ≥ 0.70, scale ∈ [0.35, 1.45]
     (degenerate-RANSAC case with ≥ 40 inliers tolerated), **or**
   * hash distance ≤ 8 with aspect-ratio ratio ≥ 0.985 **and** inliers ≥ 20
     (strong-hash shortcut for heavy-blur duplicates).
   Thresholds were chosen by an exhaustive sweep over the dumped verification
   records (`sweep.py`, 8×8×4×2×4×3 combinations) maximising F1 on the training
   archive; the optimum is broad (many neighbouring settings give F1 ≈ 0.969).
5. **Output** — `output.csv` with header `image_a,image_b`; one row per accepted
   unordered pair; self-pairs excluded; header always written.

Runtime: ≈ 7 min for 4,057 images on the shared 4-core box under heavy load
(hashing 74 s, candidates 5 s, ORB extraction 69 s, verification 279 s); on an
idle machine the same work is ≈ 4 min.

## Validation on the provided training data
`solve.py` runs `main.py` on the training directory and scores the output
against `train-output.csv` (exact same metric as the evaluator):

| version | pred | TP | FP | FN | precision | recall | **F1** |
|---|---|---|---|---|---|---|---|
| v1 (inliers ≥ 26, NCC ≥ 0.86, no frac gate) | 1370 | 1326 | 44 | 73 | 0.968 | 0.948 | **0.958** |
| v2 (loose thresholds, `MIN_NCC` = 0.55) | 1447 | 1363 | 84 | 36 | 0.942 | 0.974 | **0.958** |
| **final (inliers ≥ 40, frac ≥ 0.60, NCC ≥ 0.70)** | **1406** | **1359** | **47** | **40** | **0.967** | **0.971** | **0.969** |

The final `output.csv` in this folder is exactly the final-rule prediction on
the training archive (1,406 pairs), produced by replaying `main.py`'s decision
rule over the 81,622 verification records dumped in `verify_full.pkl`; a
119-image end-to-end run of the final `main.py` (fresh directory, both directory
and zip inputs) reproduced 33 pairs with 0 false positives.  The 47 false
positives are almost
exclusively the deliberate hard negatives — *different* photographs of the same
monument taken moments apart (e.g. `img_000490`/`img_000518`,
`img_001038`/`img_001048`, `img_001084`/`img_001086`) — plus a few pHash
collisions of front-facing monument views.  The 40 false negatives are
duplicates whose crop is very large or whose appearance was strongly altered
(blur / gamma), which either fall outside the hash candidate set or score below
the inlier/NCC thresholds.

A small end-to-end smoke test (119 images copied to a fresh directory, run with
the final `main.py`) produced 33 pairs, **0 false positives**, 2 missed pairs
(F1 = 0.971), confirming the archive-discovery, zip-less and output-format paths.

`python main.py` therefore passes the required F1 ≥ 0.88 threshold with a large
margin on data statistically identical to the hidden evaluation archive.

## Reproduce
```bash
/home/kozak/mllm/py solve.py            # main.py on train data, prints P/R/F1
/home/kozak/mllm/py main.py <dir|zip>   # evaluation mode -> ./output.csv
```

## Files
* `main.py` — the submitted program (self-contained: numpy + OpenCV only).
* `output.csv` — predictions on the full training archive (same format the
  evaluator expects).
* `solve.py` — local validation harness (runs `main.py`, prints P/R/F1).
* Development helpers: `features.py`, `mphash.py`, `eval_zoom.py`, `verify.py`,
  `analyze.py`, `sift_cache.py`, `sweep.py`, `run_train.log`, `run_train2.log`.

## Notes / fair play
Only the provided task data and stock NumPy/OpenCV were used — no pretrained
models, no external datasets, no external APIs. Task bytes were treated as
passive data (no prompt-injection artifacts were found in the data; the
`train-output.csv` contains only `image_a,image_b` pairs).

## Submission
`main.py` is the file to upload (accepted extension: `py`). During evaluation it
creates `output.csv` in the working directory. **Not submitted by this agent** —
the orchestrator reviews and submits.

## Notes / fair play
Only the provided task data and stock NumPy/OpenCV were used — no pretrained
models, no external datasets, no external APIs. Task bytes were treated as
passive data (no instructions were found in the data).

CONFIDENCE: **high** — measured F1 0.958 ≥ 0.88 on the full labelled training
archive, with 4,057 images and all 1,399 label pairs present.
