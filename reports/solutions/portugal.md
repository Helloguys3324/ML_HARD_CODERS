# Solution Report: task_portugal ("The Douro Paradox")

## 1. Overview & Objective
- **Task ID:** `task_portugal` (Portugal).
- **Difficulty & Score:** Medium, 200 / 200 points.
- **Domain:** Classic Machine Learning & Data Science / Tabular Causal Inference.
- **Output Format:** Secret flag `SIGMOID_<COLUMN_NAME>`.
- **Premise:**
  - The Douro Viticulture Institute tested a novel drip irrigation technique (`treatment`) versus conventional practices (`control`).
  - In the aggregate dataset `douro_vineyard_data.csv`, treated vineyard parcels exhibit a higher overall harvest success rate (`harvest_success`).
  - However, proper statistical stratification reveals **Simpson's Paradox**: the treatment underperforms the control within every homogeneous sub-stratum.
  - The challenge is to identify the single dataset feature that resolves the paradox and causes a full reversal of treatment effects across all groups.

---

## 2. Key Insights & Causal Analysis
- **Mechanism of Simpson's Paradox:**
  - The apparent aggregate advantage $\mathbb{P}(Y=1 \mid T=1) > \mathbb{P}(Y=1 \mid T=0)$ arises from an unconditioned confounding variable $Z$.
  - Disproportionate assignment of treatment to naturally favorable plots introduces spurious positive correlation.
  - Conditioning on confounder $Z$ reverses the effect inequality across all strata $k$:
    $$\mathbb{P}(Y=1 \mid T=1, Z=k) < \mathbb{P}(Y=1 \mid T=0, Z=k), \quad \forall k$$
- **Domain Context & Viticultural Stratification:**
  - Viticultural conditions in the Douro DOC valley vary primarily with distance inland from the Atlantic coast at Porto along the Douro river.
  - The region is historically and climatically partitioned into three subregions of increasing aridity:
    1. *Baixo Corgo:* $\le 105\text{ km}$ from Porto;
    2. *Cima Corgo:* $105\text{ km} < d \le 160\text{ km}$;
    3. *Douro Superior:* $> 160\text{ km}$.
  - The confounding column responsible for complete reversal across all zones is `distance_from_porto_km`.

---

## 3. Implementation Details
The empirical validation is implemented in `solve.py`:

1. **Dataset Loading & Preprocessing:**
   - Loads `douro_vineyard_data.csv` into a pandas DataFrame.
2. **Subregional Stratification:**
   - Partitions parcels into three discrete geographical subsets:
     ```python
     g1 = df[df["distance_from_porto_km"] <= 105]
     g2 = df[(df["distance_from_porto_km"] > 105) & (df["distance_from_porto_km"] <= 160)]
     g3 = df[df["distance_from_porto_km"] > 160]
     ```
3. **Hypothesis Verification:**
   - Evaluates conditional expectations $s_t = \mathbb{E}[\text{harvest\_success} \mid T=1]$ and $s_c = \mathbb{E}[\text{harvest\_success} \mid T=0]$ within each stratum:
     - *Baixo Corgo (g1):* $s_t < s_c$ confirmed;
     - *Cima Corgo (g2):* $s_t < s_c$ confirmed;
     - *Douro Superior (g3):* $s_t < s_c$ confirmed.
   - Comprehensive screening of alternative features (elevation, vine age, soil acidity) proved none produced full reversal across all partitions simultaneously.
4. **Flag Submission:**
   - Normalizes feature name to uppercase: `SIGMOID_DISTANCE_FROM_PORTO_KM`.
   - Submits text payload via `SigmoidClient().submit_text(...)`.

---

## 4. Results & Artifacts
- **Status:** PASSED
- **Score:** 200 / 200 points
- **Identified Confounder:** `distance_from_porto_km`
- **Recovered Flag:** `SIGMOID_DISTANCE_FROM_PORTO_KM`
- **Verification Script:** `solve.py`
