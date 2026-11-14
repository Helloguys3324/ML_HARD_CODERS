# Solution Report: task_albania ("Looks like an anomaly!")

## 1. Overview & Objective
- **Task ID:** `task_albania` (Albania).
- **Difficulty & Score:** Medium, 200 / 200 points.
- **Track:** Time-Series Anomaly Detection, Streaming ML.
- **Context:** Structural and environmental health monitoring of an underground concrete bunker in Albania via embedded sensors.
- **Input Data:** Streaming sensor readings (`train-input.csv`) ingested strictly step-by-step (`update(reading: dict)`). Six numerical features: `temperature_c`, `humidity_pct`, `vibration_g`, `acoustic_db`, `air_pressure_kpa`, `co2_ppm`, alongside `timestamp` and `bunker_zone` metadata.
- **Anomaly Types:**
  1. *Point anomalies* (sharp, instantaneous outliers);
  2. *Gradual drifts* (slow baseline level shifts);
  3. *Collective anomalies* (concurrent moderate deviations across correlated features).
- **Constraints:** Real-time processing (no lookahead, no network/external file access), standard Python runtime (`numpy`, `pandas`, `scikit-learn`), CPU execution under 60 seconds.
- **Evaluation Metric:**
  Anomaly scores $\in [0.0, 1.0]$ thresholded at $0.50$.
  $$\text{Score} = 100 \times (0.80 \times \text{overall\_F1} + 0.05 \times \text{point\_recall} + 0.075 \times \text{drift\_recall} + 0.075 \times \text{collective\_recall})$$
  Passing threshold: $\text{Score} \ge 70.00 / 100$.

---

## 2. Key Insights & Methodology
The solution employs a lightweight, robust streaming statistical framework without neural network overhead:
1. **Dual Sliding-Window Architecture:**
   - **Long Window ($W_{\text{long}} = 720$ steps):** Computes robust baseline statistics using rolling median and Median Absolute Deviation ($\text{MAD}$, with normal scale $\sigma \approx 1.4826 \times \text{MAD}$).
   - **Short Window ($W_{\text{short}} = 30$ steps):** Tracks instantaneous local level shifts for rapid drift detection.
2. **Variance Floor (`SIGMA_FLOOR`):** Protects against zero-division and degenerate variance during warm-up or low-noise regimes (`temperature_c: 0.10`, `humidity_pct: 0.20`, `vibration_g: 0.003`, `acoustic_db: 0.10`, `air_pressure_kpa: 0.02`, `co2_ppm: 3.0`).
3. **Multi-Aspect Deviation Aggregation:**
   - Evaluates instantaneous deviation $z_{\text{inst}}[f] = \frac{x_f - \text{med}_f}{\sigma_f}$ and short-term level shift $z_{\text{level}}[f] = \frac{\text{median}(\text{short}_f) - \text{med}_f}{\sigma_f}$ before appending the new point (preventing self-leakage).
   - Combines single-feature maximums ($spike = \max_f |z_{\text{inst}}[f]|$, $lev = \max_f |z_{\text{level}}[f]|$) with root-mean-square of top-3 deviating features ($multi = \sqrt{\frac{1}{3}\sum_{k=1}^3 (top_k(event))^2}$) to catch collective anomalies.
4. **Calibrated Probability Mapping:**
   Maps raw deviation $raw = \max(spike, lev, multi)$ to $[0.0, 1.0]$ via exponential CDF:
   $$\text{score} = 1.0 - \exp\left(-\frac{\max(raw, 0)}{4.3}\right)$$
   This aligns the $0.50$ decision boundary with an approximate $3.0\sigma$ deviation.

---

## 3. Step-by-Step Implementation Details

### Core Scripts (`task_albania/`)
- **`detector.py` (Submission Module):**
  - Implements `Detector.update(self, reading: dict) -> float`.
  - Buffers history using `collections.deque(maxlen=720)` and `deque(maxlen=30)`.
  - Returns `0.0` during the initial warm-up phase ($n < 40$).
  - Evaluates deviations, updates circular buffers, computes composite metric $raw$, and applies clipped probability mapping.
- **`solve.py` (Local Evaluation & Verification Harness):**
  - Simulates streaming ingestion from `train-input.csv`.
  - Computes confusion matrix ($TP, FP, FN, TN$), precision, recall, overall $F_1$, and specific recalls (`r_point`, `r_drift`, `r_coll`).
  - Verifies that final competition score exceeds $70.00$.

---

## 4. Results & Artifacts
- **Status:** `PASSED` (Score: **200 / 200** points).
- **Submission Attempt:** Attempt 1 (`2026-11-14T11:20:37.696Z`, submitter: `Veaceslav C`).
- **Detection Performance:**
  - Robust against point spikes, slow drifts, and multi-sensor collective events.
  - Zero false positives during initial initialization window.
- **Flag:**
  ```text
  SIGMOID_BUNKER
  ```
- **Key Artifacts:**
  - `detector.py` (4,287 bytes, standalone detector class);
  - `solve.py` (streaming verification harness).
