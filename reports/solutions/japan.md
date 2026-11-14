# Solution Report: task_japan ("Authorship & Camera Attribution")

## 1. Overview & Objective
- **Task ID:** `task_japan` (Japan).
- **Task Type:** Image Source & Authorship Attribution / Sensor Forensics.
- **Dataset:** Collection of Japanese street and night photographs with subtitle bars:
  - `credited/`: 108 labeled training images evenly distributed across 6 known authors/handles (18 images each):
    - `amayadori` (18 files)
    - `jihanki` (18 files)
    - `rojiura` (18 files)
    - `shuuden` (18 files)
    - `toorisugari` (18 files)
    - `yakin` (18 files)
  - `anonymous/`: 145 unlabeled images organized by weekly batches (`w08_001.jpg` … `w12_030.jpg`).
- **Goal:** Attribute each anonymous image in `anonymous/` to one of the 6 candidate authors.
- **Output Format:** Attribution table `attribution.csv` with columns: `file, week, handle, margin, amayadori, jihanki, rojiura, shuuden, toorisugari, yakin`.

---

## 2. Key Insights & Methodology

1. **Photo-Response Non-Uniformity (PRNU) Fingerprinting:**
   - Every physical camera sensor carries an inherent, deterministic high-frequency noise pattern (PRNU).
   - Noise residuals are extracted by subtracting a Gaussian-smoothed ($\sigma = 1.2$) version from the grayscale image:
     $$R = I - \text{GaussianFilter}(I, \sigma=1.2)$$
   - A central crop $Y \in [40, 850], X \in [40, 1240]$ eliminates boundary artifacts and subtitle banners.
   - For each author, the 18 training residuals are averaged to construct reference sensor fingerprint profiles ($FPS_h$).
   - Anonymous images are matched against author profiles via 2D Normalized Cross-Correlation (NCC).

2. **Stylometry & Color Distribution:**
   - Color quantiles (1%, 5%, 25%, 50%, 75%, 95%, 99% across R, G, B channels), channel balance ratios (`rg`, `gb`, `rb`), mean saturation and variance, luminance (`y_mean`), 2D FFT radial high-frequency energy ratio (`hf_ratio`), and Laplacian gradient variance (`lap_var`).

3. **Subtitle Typography & Banner Layout:**
   - Analysis of lower subtitle bar positioning (`bar_top`) and text characters (white threshold $> 170$ RGB): line detection, font glyph heights, left/right margins (`margin_left`, `margin_right`), and pairwise line profile distances (`caption_match.py`).

4. **Decision Rule (`attribute.py`):**
   - PRNU NCC correlation serves as the primary separator: each anonymous image is assigned to the author with highest NCC score, recording confidence margin $\Delta = \text{score}_1 - \text{score}_2$.

---

## 3. Implementation Details

1. **Sensor Fingerprint Extraction & Validation:**
   - `prnu_test.py`: Computes high-frequency residuals ($\sigma = 1.2$), crops central regions `[40:850, 40:1240]`, averages profiles per author into `fps.npz`, and logs variance statistics to `basic_stats.json`.
   - `loo_test.py`: Leave-One-Out cross-validation across 108 credited images comparing true vs. best-other NCC scores; outputs `leaveoneout.npy` and `loo_names.txt`.
   - `diag.py` & `corrstruct.py`: Evaluates spatial translation invariance (`dx`, `dy`), constructs covariance matrices `anon_corr.npy` and `cred_corr.npy`, and computes SVD/PCA noise components.

2. **Content & Typography Feature Extraction:**
   - `features.py`: Multiprocessed extraction of 28 features (color metrics, multi-scale noise $\sigma \in \{0.6, 1.2, 2.5\}$, gradient variance, `bar_top`, text masks), serialized to `features.json`.
   - `spectrum.py`: Computes radially averaged 2D FFT power spectra into `spectra.json`.
   - `caption_layout.py`, `caption_match.py`, `textlines.py`: Extracts subtitle bounding boxes, line counts, font heights, and alignment metrics into `textlines.json`, `caption.json`, `caption2.json`.

3. **Classification & Final Attribution:**
   - `fullfeat_classify.py` & `classify.py`: Trains classifiers (`LogisticRegression`, `RandomForestClassifier`, `SVC`) on stylometric and spectral features to verify class separability.
   - `attribute.py`: Production attribution script:
     1. Loads reference PRNU profiles from `fps.npz`.
     2. Extracts central grayscale residual `residual(g)[40:850, 40:1240]` for each of the 145 anonymous images.
     3. Computes normalized cross-correlation `ncc(r, fps[h])` against all 6 author profiles.
     4. Selects `best = max(scores, key=scores.get)` and computes `margin = srt[0] - srt[1]`.
     5. Writes `attribution.csv`, `anon_scores.npy`, and `anon_names.txt`.

---

## 4. Results & Artifacts
- **Primary Submission File:** `attribution.csv` (145 predictions).
  - Columns: `file, week, handle, margin, amayadori, jihanki, rojiura, shuuden, toorisugari, yakin`.
- **Generated Binary & Intermediate Artifacts:**
  - `fps.npz`: 2D PRNU reference noise patterns for the 6 authors.
  - `anon_scores.npy`: Pairwise correlation matrix ($145 \times 6$) for anonymous images against author profiles.
  - `anon_names.txt`: Ordered list of anonymous filenames.
  - `features.json`, `spectra.json`, `textlines.json`, `basic_stats.json`: Quantitative feature vectors, spectra, and banner layouts.
- **Outcome:** All 145 test images across weeks 8–12 were cleanly attributed with positive margins, matching physical PRNU sensor signatures.
