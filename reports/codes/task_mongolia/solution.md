# task_mongolia — "The Baa-rilliant task!" (Medium, 200 pts, text answer)

**Answer candidate:** `SIGMOID_SHEEP`
**CONFIDENCE: low** — the per-plot sheep counts were estimated successfully
(MAE ≈ 8 on held-out training flocks), the plot grid was fully recovered
(37×9 lattice), but the final census→word decode was **not** cracked. The
candidate above is a thematic guess, not a verified decode. Alternative
candidates are listed in §6. Do not treat this as a verified flag.

---

## 1. Data

`work/tasks/task_mongolia/data/` (intake, 2026-11-14):

| file | content |
|---|---|
| `train/train_000.jpg … train_279.jpg` | 280 drone photos, 600×600 RGB |
| `train_counts.csv` | exact sheep count per train image (0–119, mean 41.5) |
| `train_points.csv` | 11 629 body-centre dots (one per sheep, x/y in pixels) |
| `survey/survey_0001.jpg … survey_0333.jpg` | 333 survey photos (test) |
| `survey_plots.csv` | flight log: `image,x_m,y_m` (plot position in metres) |

Integrity checks: point count per image equals `train_counts.csv` exactly for all
280 images (11 629 = 11 629). No EXIF/GPS, no JPEG comment segments, no hidden
files; all images 600×600 JFIF. Nothing prompt-injection-like in the data.

## 2. Plot grid recovery (Observed)

The 333 survey photos are **not** a random scatter: `survey_plots.csv` is a
jittered rectangular lattice. Working from pairwise displacement vectors of
nearest neighbours (cKDTree, r < 52 m) the two lattice basis vectors are

```
u = (44.92, 22.36) m      v = (22.46, -44.19) m      |u|≈50.2 m, |v|≈49.8 m
```

i.e. an orthogonal 50 m grid rotated ≈ 26.6° in the survey frame. Expressing
every plot in this basis gives residuals < 0.37 cell and a **complete 37×9 grid
(37×9 = 333, every cell filled exactly once)**. Files:
`survey_plots_ij.csv`, `plot_ij.npy`, `plots_positions.png`.

This 37×9 canvas is the strongest clue that the census is meant to be *drawn*:
37 = 6×5 columns + 5 one-pixel gaps + 2 margins, 9 = 7 rows + 2 margins — i.e.
exactly a **6-letter word in a 5×7 dot-matrix font**.

## 3. Counting sheep (method + validation)

No pretrained models / external data were used (fair-play default). Three
from-scratch classical approaches were built and compared on a held-out split of
the 280 annotated training photos:

1. **Global image features + LightGBM regression** (`global_reg.py`)
   — 46 features: whiteness (`V·(1−S)^1.2`) fractions at 5 thresholds, HSV/Lab
   statistics, white top-hat areas at 3 scales, connected-component counts, DoG
   local-maxima counts at 6 thresholds, black top-hat (dark sheep) areas.
   5-fold CV: **MAE = 8.2 sheep, corr = 0.906** (Ridge: MAE 11.7).
2. **Cell-level density regression** (`cell_density.py` + `cell_reg_fast.py`)
   — the image is cut into 60×60 cells (10×10); each cell gets 21 local
   features (mean/std whiteness, top-hat area fractions, per-cell DoG peak
   counts, black-hat area, colour stats); LightGBM regresses the number of
   annotated centres per cell and the per-image count is the sum.
   In-sample image MAE 7.5.
3. Rejected baselines (kept for the record): LoG/DoG blob counting
   (`classical.py`, `dog_baseline.py` — MAE ≫ 50, gross over-detection on
   grass texture); single average sheep template (`template_match.py`,
   best F1 0.38); 20 real-patch template bank (`tpl_bank.py`, corr 0.39).

Final survey counts = 0.5·(global) + 0.5·(cell), rounded, clipped ≥ 0:
`counts.csv` (333 rows), `make_counts.py`. Survey distribution: mean 35.1,
median 33, range 3–132, **total ≈ 11 693 sheep**.

Known bias (documented): the regressors shrink towards the training mean, so
very small flocks are over-estimated (e.g. `survey_0123.jpg` predicted 1.2 but
≈8–10 sheep are visible) and very large flocks are under-estimated.

## 4. Census decode attempts (all failed)

Using the predicted 37×9 count grid (rendered `gray_raw.png`, `cellgrid_*`,
`grid_*`):

* Grayscale / residual (row+col, polynomial, Gaussian background removal)
  renders at every percentile threshold 20–85 %: **no readable letters**.
* 5×7 dot-matrix font scoring on all 8 grid symmetries, layouts of 6/5/4
  letters (1-px and 2-px spacing), bright- and dark-glyph polarity, and three
  background models (`font_decode.py`, `symmetry_decode.py`). Best layout is
  x = 1,7,13,19,25,31 (consistent with the 37×9 canvas), but the best-scoring
  letters are unstable (`REQAMB`/`REPALL`/`RECHND`, score margins ≈ noise).
* Binary-threshold → ASCII (5/6/7/8-bit), Bacon/A1Z26, Braille (3×2 cells),
  count+offset ASCII (offsets 0/32/48/64/65), row/column sums mod 26,
  rank/morse/serpentine/column-major/spiral orders: no intelligible text.
* 3×5 micro-font layouts (7 letters): garbage.
* Decimal digits of `x_m,y_m` (jitter): uniform random (χ²-consistent), no
  message. `survey_plots.csv` contains only the 3 documented columns.
* No survey/train near-duplicate images (aHash Hamming distance ≥ 30).

**Diagnosis:** the 37×9 canvas demands a 6-letter word, but the count field we
can recover is too noisy (MAE ≈ 8, variance shrunk) to resolve glyphs whose
contrast is comparable to the noise. The intended decode is almost certainly a
dot-matrix rendering of a 6-letter word in the per-plot counts; a near-exact
counter (human verification or a proper detector) is required to read it.

## 5. Evidence that counting is otherwise sound

`train_samples.png` overlays the annotated points on train images;
`lowest_pred.png` shows the 15 lowest-predicted survey photos — all do contain
sheep, so the survey has no empty "background" plots (presence/absence bitmaps
are ruled out; the counts look like a natural flock-size distribution,
unimodal around ≈35, not a two-level bitmap).

## 6. Flag candidates / recommendation

| candidate | basis |
|---|---|
| **`SIGMOID_SHEEP`** | thematic (the census counts sheep); submitted as best guess |
| `SIGMOID_CENSUS` | 6 letters, exact 37×9=6×5+5+2 font fit, self-referential |
| `SIGMOID_HERDER` / `SIGMOID_FLOCKS` | 6-letter story words |
| `SIGMOID_REPORT` | only word consistent with the weak font signal `REP…` |
| `SIGMOID_SUREG` | Mongolian *сүрэг* = flock/herd; team flags for country tasks are often local-language words (cf. `SIGMOID_SUMU`, `SIGMOID_SAWUBONA`, `SIGMOID_NGIYAKUZWA`) |

**Next action for the orchestrator:** do not spend a submission on this guess
unless nothing better is available. The high-value continuation is to (a)
manually verify the counts of ~40–60 plots (or run the pixel-level LightGBM
classifier `pixel_clf.py`, which needs ~40 min under current CPU load), then
(b) re-render the 37×9 grid and read the 6-letter 5×7 glyphs directly. All
scripts and the tuned plots are in this directory.

## 7. Files

`answer.txt`, `counts.csv` (333 rows), `solution.md`, `solve.py`,
`train_unet.py`, `pixel_clf.py`, `global_reg.py`, `cell_density.py`,
`cell_reg_fast.py`, `make_counts.py`, `font_decode.py`, `symmetry_decode.py`,
`classical.py`, `dog_baseline.py`, `template_match.py`, `tpl_bank.py`,
`render_grid.py`, `explore_grid.py`, `survey_plots_ij.csv`,
`survey_pred_global.csv`, `survey_pred_cell.csv`, `survey_montage.png`,
`train_samples.png`, `lowest_pred.png`, `plots_positions.png`.
