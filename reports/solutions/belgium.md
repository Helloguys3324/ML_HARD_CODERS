# Solution Report: task_belgium ("Fry-ndship")

## 1. Overview & Objective
- **Task ID:** `task_belgium` (Belgium).
- **Difficulty & Score:** Easy, 100 / 100 points.
- **Track:** Tabular Regression, Data Steganography.
- **Context:** A Flemish potato farming cooperative in Belgium fencing its agricultural parcels based on surveyor measurements.
- **Input Data:** `potato_fields.csv` (3,869 rows) with features:
  - Identifiers & geography: `field_id`, `crop_code`, `crop`, `farm_type`, `lat`, `lon`;
  - Geometry: `area_ha`, `long_side_m`, `ditch_m`;
  - Field observations: surveyor Marietta's hedge length `hedge_m`, actual perimeter/boundary `edge_m`, and contractor quote `fence_quote_eur`.
- **Premise:** A regression model explains 99% of boundary variance from raw parcel dimensions, yet surveyor Marietta's field measurements encode a hidden visual watermark.
- **Objective:** Model boundary dimensions and discover the steganographic word hidden within survey residuals.
- **Reward:** 100 points, Antarctica meta-quest Slot 3 clue (`belongs`), and flag `SIGMOID_<WORD>`.

---

## 2. Key Insights & Methodology

### 1. Geometric Boundary Modeling ($R^2 = 0.99$)
Assuming rectangular fields, the short side is computed as:
$$S = \frac{\text{area\_ha} \times 10{,}000}{\text{long\_side\_m}}$$
The theoretical perimeter is:
$$\text{perim} = 2 \times (L + S)$$
Linear regression of `edge_m` against field dimensions yields $R^2 = 0.9890$, confirming the declared 99% explanatory power. The empirical measurements strictly obey linear scaling against the true geometric perimeter:
$$\text{edge\_m} \approx 0.95 \times \text{perim}$$
$$\text{hedge\_m} \approx 0.04 \times \text{perim}$$

### 2. Scatter Plot Residual Steganography
Subtracting the deterministic linear trends reveals point-based scatter steganography:
- Plotting boundary residuals $\text{res\_edge} = \text{edge\_m} - 0.95 \times \text{perim}$ against `perim` renders the clear textual banner: **`REGRESSION`**.
- Plotting hedge residuals $\text{res\_hedge} = \text{hedge\_m} - 0.04 \times \text{perim}$ against `perim` renders the target keyword: **`SUPERVISION`**.

The secret keyword is **`SUPERVISION`**, yielding flag `SIGMOID_SUPERVISION`.

---

## 3. Step-by-Step Implementation Details

### Implementation Script (`solve.py`)
1. **Geometric Feature Engineering:**
   ```python
   area_m2 = df["area_ha"] * 10000.0
   short_side_m = area_m2 / df["long_side_m"]
   perim = 2.0 * (df["long_side_m"] + short_side_m)
   df["perim"] = perim
   ```
2. **Residual Extraction:**
   ```python
   df["res_edge"] = df["edge_m"] - 0.95 * perim
   df["res_hedge"] = df["hedge_m"] - 0.04 * perim
   ```
3. **Artifact Visualization:**
   Saves scatter plots to `res_edge_regression.png` and `res_hedge_supervision.png` using `matplotlib.pyplot.scatter`.
4. **Platform Submission:**
   Submits flag `SIGMOID_SUPERVISION` via `SigmoidClient.submit_text("task_belgium", flag, url=repo_url)`.

---

## 4. Results & Artifacts
- **Status:** `PASSED` (Score: **100 / 100** points).
- **Submission Attempt:** Attempt 3 (`2026-11-14T10:08:20.306Z`, submitter: `Kozak`).
- **Antarctica Quest Clue Unlocked:** `belongs` (Slot 3).
- **Recovered Flag:**
  ```text
  SIGMOID_SUPERVISION
  ```
- **Key Artifacts:**
  - `res_edge_regression.png` (visual confirmation of "REGRESSION");
  - `res_hedge_supervision.png` (visual confirmation of "SUPERVISION");
  - `solve.py` (3,223 bytes);
  - `uniform_tests.csv` (statistical verification metrics).
