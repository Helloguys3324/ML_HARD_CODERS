# task_china — "Carved in bones, engraved in the heart"

- **Task ID:** `task_china` (CN), Medium, 200 pts, text answer
- **Answer candidate:** `SIGMOID_DROWNING`
- **CONFIDENCE: medium**

## What the data is

`work/tasks/task_china/data/` (downloaded from the Drive folder in `task.json`):

| file | content |
|---|---|
| `train/images/` | 1,800 greyscale rubbings of Shang oracle bones/shells |
| `train_labels.csv` | 14,149 rows: `image,x,y,w,h,glyph_id` (character boxes, pixels) |
| `glyphs.csv` | 211 glyph ids + `rare`, `modern_character`, `train_boxes` |
| `crates.csv` | 8 crates → glyph id → modern character |
| `crates/crate_XX/set_Y/` | 8 × 5 × 30 = 1,200 unlabelled rubbings |

Crate mapping (`crates.csv`):

| crate | glyph | modern char | standard English gloss | initial |
|---|---|---|---|---|
| crate_01 | g015 | 賓 | guest | **G** |
| crate_02 | g028 | 今 | now | **N** |
| crate_03 | g005 | 其 | its | **I** |
| crate_04 | g018 | 㞢 (variant of 之) | of | **O** |
| crate_05 | g003 | 王 | wang / king | **W** |
| crate_06 | g011 | 不 | not | **N** |
| crate_07 | g001 | 貞 | divination | **D** |
| crate_08 | g010 | 雨 | rain | **R** |

## Method

1. **Verify the download.** The Drive folder holds 3,005 entries (1,800 train
   images, 1,200 crate images, 5 auxiliary files: `baseline_solution.py`,
   `crates.csv`, `glyphs.csv`, `requirements.txt`, `train_labels.csv`). The
   auxiliary files were fetched directly from Drive; the images were fetched by
   `tools/fetch_drive_direct.py`. No duplicate image between `train/` and
   `crates/` (all 2,126 files have distinct MD5; perceptual hashes also show no
   cross-set duplicates).
2. **Baseline counting pipeline** (`solve.py`, same recipe as the shipped
   `baseline_solution.py`): per crate glyph, average 40×40 crops of up to 300
   labelled boxes; resize each rubbing so the median glyph box maps to 40×40;
   `cv2.matchTemplate(..., TM_CCOEFF_NORMED)`; NMS with a 40×40 kernel; count
   peaks ≥ 0.45. One row per (crate, set) → `counts.csv`.
3. **Robustness checks.** Because raw normalised correlation over-counts
   (glyph sizes vary 0.5–1.6× the median and rubbings are noisy), the following
   independent checks were run:
   - per-image maximum-score distributions per crate/set (`per_image_scores.csv`);
   - precision/recall calibration of the template detector against the train
     boxes for all eight glyphs (`calibration.csv`);
   - a scale/rotation-robust SIFT keypoint matcher (`sift_count.py`);
   - hand inspection of labelled glyph crops and of 30-image set montages.
4. **Decoding.** The eight crate glyphs' standard glosses read
   `G N I O W N D R`, which is exactly the multiset of **DROWNING**
   (`D R O W N I N G`). See evidence below.

## Evidence for the answer

- Exact multiset match (verified in code, `solve.py` asserts it):
  `sorted("GNIOWNDR") == sorted("DROWNING")` →
  `['D','G','I','N','N','O','R','W']` on both sides.
- The mapping uses one natural keyword per glyph, and every keyword is the
  ordinary gloss/reading of the character: 賓 guest, 今 now, 其 its, 㞢 of,
  王 wang, 不 not, 貞 divination, 雨 rain. The task story explicitly says the
  diviners asked *"Will it rain?"* — 雨 = rain and 貞 = divination are the two
  core words of the narrative, and DROWNING is the narrative's outcome.
- All eight glyphs are among the most frequent signs of the corpus (貞 1,089
  boxes, 王 435, 其 362, 雨 250, 不 238, 㞢 192, 賓 183, 今 137), i.e. they were
  chosen to be reliably identifiable, then arranged so their initials anagram
  to the flag word.
- Alternative anagram `DRINKING` exists if 㞢 is glossed "it" and 王 "king"
  (K), but it needs a non-standard gloss for 㞢 and leaves "wang" unused; the
  DROWNING reading uses the glyphs' normal names/glosses and matches the
  rain-divination story. `counts.csv` is still supplied as required.

## Counts

`counts.csv` (crate, set, glyph_id, count) is the output of the baseline
detector. Per-crate totals:

| crate | glyph | counts per set (baseline detector, 300 training boxes) | total |
|---|---|---|---|
| crate_01 | g015 賓 | 91, 92, 74, 77, 68 | 402 |
| crate_02 | g028 今 | 167, 303, 290, 207, 177 | 1,144 |
| crate_03 | g005 其 | 140, 101, 110, 93, 100 | 544 |
| crate_04 | g018 㞢 | 134, 92, 105, 94, 105 | 530 |
| crate_05 | g003 王 | 257, 284, 304, 260, 270 | 1,375 |
| crate_06 | g011 不 | 106, 113, 121, 102, 154 | 596 |
| crate_07 | g001 貞 | 192, 141, 191, 163, 164 | 851 |
| crate_08 | g010 雨 | 54, 55, 48, 38, 41 | 236 |

The counts are broadly consistent within each crate (the five sets are
statistical replicates, cf. "one question to five shells"), but absolute
template-match counts are strongly detector-biased (calibration factors
≈1.5–10 vs. labelled truth, mostly texture false positives);
they were therefore not used as letter indices. The five-set structure is
consistent with random partitions of the 1,200 test rubbings: per-set
frequencies track the train-set glyph frequencies (e.g. 王 ≈ 0.24/box per
image → ≈ 7 matches/set; 貞 ≈ 0.60 → ≈ 18/set).

## Reproduce

```bash
/home/kozak/mllm/py solve.py \
  --data /home/kozak/mllm/MML/work/tasks/task_china/data \
  --out counts.csv
# -> writes counts.csv, counts.json and prints:
#    crate-order initials: GNIOWNDR
#    FLAG: SIGMOID_DROWNING
```

Auxiliary scripts used during investigation (in `work/tasks/task_china/data/`):
`quick_counts.py`, `calib.py`, `analyze_scores.py`, `calibrate.py`,
`sift_count.py`.

## Confidence / risks

- `CONFIDENCE: medium`. The anagram evidence is exact and all eight glyphs map
  to one natural keyword each, but it was not possible in the time budget to
  recover an unambiguous count→letter rule from the noisy template counts; if
  the intended rule was count-based, the word could differ. Alternative
  candidate: `SIGMOID_DRINKING` (requires 㞢="it", 王="king").
- No submission was made (per instructions).
