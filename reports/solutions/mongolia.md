# Solution Report: task_mongolia ("The Baa-rilliant task!")

## 1. Overview & Objective
- **Task ID:** `task_mongolia` (Mongolia).
- **Difficulty & Value:** Medium (200 points); submission via text answer (`answer.txt`).
- **Dataset:**
  - Training Set: 280 drone aerial images of pasture land (`train/train_000.jpg` … `train_279.jpg`), $600 \times 600$ RGB.
  - `train_counts.csv`: Ground-truth sheep counts per training frame (0 to 119, mean 41.5).
  - `train_points.csv`: Pixel coordinates of sheep centers (11,629 annotated points; sums match `train_counts.csv` exactly).
  - Test Set: 333 aerial images (`survey/survey_0001.jpg` … `survey_0333.jpg`), $600 \times 600$.
  - `survey_plots.csv`: Flight telemetry log with spatial center coordinates in meters (`image, x_m, y_m`).
- **Objective:** Count sheep across all 333 survey plots, reconstruct the flight survey grid topology, and decode the hidden keyword/flag in format `SIGMOID_<WORD>`.

---

## 2. Key Insights & Methodology

1. **Flight Lattice Recovery & Geometry:**
   - Coordinates $(x, y)$ in `survey_plots.csv` follow a regular rectangular grid subject to slight GPS jitter.
   - Nearest-neighbor displacement analysis via `scipy.spatial.cKDTree` ($r < 52$ m) identified two orthogonal basis vectors:
     $$u = (44.92, 22.36)\text{ m}, \quad v = (22.46, -44.19)\text{ m} \quad (|u| \approx 50.2\text{ m}, |v| \approx 49.8\text{ m})$$
     corresponding to an orthogonal 50-meter lattice rotated by $\approx 26.6^\circ$.
   - Decomposing plot coordinates into this basis with integer rounding $(i, j)$ produces an exact $37 \times 9$ lattice:
     - Exactly 333 plots, zero collisions, each cell occupied once (residual $< 0.37$).
   - The $37 \times 9$ canvas strongly suggests a 6-letter word rendered in a standard $5 \times 7$ dot-matrix font:
     $$37 = 6 \times 5\text{ (glyph width)} + 5\text{ (letter spacing)} + 2\text{ (horizontal margins)}$$
     $$9 = 7\text{ (glyph height)} + 2\text{ (vertical margins)}$$

2. **Fair-Play Classical ML Sheep Counting:**
   - Designed a lightweight two-model ensemble trained from scratch without external pre-trained models:
     1. **Global Frame Regression (`global_reg.py`):** 46 global features including whiteness ratios $V \cdot (1 - S)^{1.2}$ across 5 thresholds, HSV/Lab channel statistics, white morphological top-hat filters across 3 scales, Difference-of-Gaussians (DoG) peak counts across 6 thresholds, and black top-hat filters for dark sheep. 5-fold CV: **MAE = 8.2 sheep, Pearson $r = 0.906$**.
     2. **Local Cell Density Regression (`cell_density.py` + `cell_reg_fast.py`):** Partitions each frame into a $10 \times 10$ grid of $60 \times 60$ px patches. Extracts 21 local features per patch; trains LightGBM to predict cell-level sheep counts. Summing all 100 cells yields total frame count (**MAE = 7.5**).
     3. **Ensemble:** Blends predictions via $\text{count} = \text{round}(0.5 \cdot \text{pred}_{\text{global}} + 0.5 \cdot \text{pred}_{\text{cell}})$. Total estimated sheep across 333 plots: $\approx 11,693$ (mean 35.1, median 33, range 3 to 132).

3. **Decoding Bottleneck:**
   - Evaluated the $37 \times 9$ count matrix against $5 \times 7$ raster fonts across 8 symmetry groups (`font_decode.py`, `symmetry_decode.py`), 2D spatial detrending, percentile thresholding, and alternate encodings.
   - Regression noise (MAE $\approx 8$) and regression-to-the-mean shrinkage blur glyph edges, preventing statistically unambiguous automated OCR.

---

## 3. Implementation Details

1. **Lattice & Pipeline Orchestration (`solve.py`):**
   - `fit_lattice(csv_path)`: Builds spatial k-d tree, clusters neighbor vectors, solves for basis $(u, v)$, maps coordinates to discrete $(i, j)$, and saves `survey_plots_ij.csv`.
   - Manages feature extraction and merges predictions into `counts.csv`.

2. **Model Training & Prediction:**
   - `global_reg.py`: Extracts 46 global color and morphological features; generates `survey_pred_global.csv`.
   - `cell_density.py` & `cell_reg_fast.py`: Extracts patch-level features and runs fast regression, yielding `survey_pred_cell.csv`.
   - `make_counts.py`: Merges and formats predictions.

3. **Decoding & Visualization Modules:**
   - `font_decode.py` & `symmetry_decode.py`: $5 \times 7$ raster font matcher across spatial alignments.
   - `render_grid.py`, `explore_grid.py`: Renders grayscale 2D heatmaps of density distributions.

---

## 4. Results & Artifacts
- **Output Tables:**
  - `counts.csv`: Predicted sheep counts for all 333 survey plots (`image, count`).
  - `survey_plots_ij.csv`: Lattice mapping of survey images to $37 \times 9$ coordinates.
  - `survey_pred_global.csv`, `survey_pred_cell.csv`: Individual model predictions.
- **Flag Hypothesis & Assessment:**
  - `answer.txt`: Contains domain hypothesis `SIGMOID_SHEEP`.
  - **Confidence: LOW.** Glyphs could not be decoded above the regression noise floor.
  - Plausible domain candidates identified:
    - `SIGMOID_CENSUS` (exact 6-letter match for the $37 \times 9$ layout representing a sheep census).
    - `SIGMOID_REPORT` (corresponds to weak font-matching responses `REP…`).
    - `SIGMOID_SUREG` (Mongolian *süreg* = herd/flock).
