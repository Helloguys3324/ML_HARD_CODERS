# Solution Report: task_france ("Point par point")

## 1. Problem Overview
- **Task & Track:** Easy (100 pts). The challenge references Georges Seurat, Paul Signac, and the 1886 Parisian Pointillist movement. A dataset of Paris Airbnb listings (`paris_listings.csv`, 28,108 rows, 18 attributes) supposedly contains injected coordinate points whose "colors correspond to what travelers evaluate before booking" (pricing/financial fields). A clue states: *"Grey is pretty. It holds multiple secrets"*.
- **Objectives:**
  1. Detect injected coordinate points spelling out letters across Paris.
  2. Read the hidden word in authorial order.
  3. Produce audit/clustering code, diagnostic visualizations, and `labels.csv`.
  4. Submit flag `SIGMOID_<WORD>`.

## 2. Forensic Audit & Spatial Analysis
A comprehensive statistical and spatial audit was executed in `solve.py` and documented in `solution.md`:
1. **Metadata & Coordinate Forensics:**
   - `listing_id` is uniformly distributed in $[1\,000\,358, 9\,999\,630]$, indicating synthetic generation without payload.
   - Coordinates (`latitude`, `longitude`) are bounded strictly within Paris (48.816–48.902 N, 2.234–2.465 E) rounded to 5 decimal places (90.4% at 5 decimals, 9.3% at 4 decimals); no mantissa steganographic bitplanes were detected.
   - Price fractions follow exact divisibility by `minimum_nights` ($k/2, k/3, k/4, k/5, k/7$) with an additional $k/11$ hotel subset. Exactly 7 listings have `calculated_host_listings_count == 225` (6 rounded to 3 decimal places), but their coordinates are dispersed across opposite sides of the city and cannot form letters.
2. **Spatial Glyph Search:**
   - Over 40 filters were tested: grayscale price/rating projections, local density contrast, DBSCAN ($\varepsilon \in [30, 200]\,\text{m}$), $k$-NN distance thresholds ($d_{31}$, $d_{200}$), 2D Fourier lattice scans, and oriented matched filters ($300 \times 80\,\text{m}$ and $150 \times 40\,\text{m}$ across 12 angles).
   - Point densities exclusively match true Parisian residential architecture without isolated glyph geometry.
3. **Hypothesis Formation:**
   - Given the lack of confirmed geometric glyphs in spatial clusters, the primary contextual candidate derived from the task statement and pointillist history is **SEURAT** (Georges Seurat, 6 letters).

## 3. Implementation Details (`solve.py`)
- **Metric Coordinate Projection:**
  Converts WGS84 coordinates to local metric offsets relative to Paris center (`LAT0 = 48.8566`, `LON0 = 2.3522`):
  $$x = (\text{lon} - \text{LON0}) \cdot 111320 \cdot \cos(\text{LAT0}), \quad y = (\text{lat} - \text{LAT0}) \cdot 111320$$
- **Clustering & Exports:**
  - `DBSCAN(eps=200, min_samples=5)` on metric coordinates.
  - Generates `labels.csv` (28,108 rows) containing `listing_id` and `dbscan_cluster`.
  - Saves scatter plots to `figs/` (`01_all_points.png` through `05_isolated.png`).

## 4. Results & Artifacts
- **Candidate Flag:** `SIGMOID_SEURAT` (Confidence: Low — derived from historical narrative context after exhaustive spatial audit).
- **Files:** `solve.py`, `solution.md`, `labels.csv` (309 KB), `answer.txt`.
