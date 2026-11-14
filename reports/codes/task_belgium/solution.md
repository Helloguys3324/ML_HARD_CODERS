# task_belgium — "Fry-ndship" (Easy, 100 pts, text flag)

**Answer candidate:** `SIGMOID_UNIFORM`
**CONFIDENCE: medium** (see "Uncertainty / alternatives" at the end)

## TL;DR

1. The analyst's model of the story is the regression

   `edge_m ~ area_ha + long_side_m`  ("from their dimensions")

   which gives **R² = 0.9890** — exactly the *"model explained 99% of the
   variation in edge length"* of the statement.
2. Writing `L = long_side_m` and `S = area_ha*10000/L` (short side), the
   measurements obey an exact linear law

   `edge = 0.82·(L+S) + ditch + hedge + ε`,  `|ε| < 0.107 m`,  `σ(ε)=0.058 m`

   i.e. `edge ≈ 1.9·(L+S)` = 95 % of the rectangle perimeter.
3. The designed, non-generic property of the surveyed fields is the
   distribution of **D = L + S** (equivalently the measured edge length):
   **D is uniformly distributed** over its support.
   L and S are *not* uniform (they are bell/right-skewed), L and S are
   **positively** correlated (ρ = +0.358), and yet their sum is flat
   (skew −0.03, excess kurtosis −1.13, close to the uniform value −1.2).
   That cannot happen for a generic sum of two positive-correlated
   variables. In fact `f = L/D` is ≈ Normal(0.72, 0.09) and `D` is
   ≈ Uniform(133, 584) with `L = f·D`, `S = (1−f)·D`.
4. Therefore the measured edge lengths are spread **uniformly** across
   their range — Mariette's "oddly organised" sample is a **uniform /
   evenly spread** sample of field sizes.

**Word: UNIFORM → `SIGMOID_UNIFORM`**

## Evidence

### 1. The 99 % model

```
edge_m ~ area_ha + long_side_m
R² = 0.9890
coef = [60.7125, 1.4386], intercept = 140.95
```

This matches the story ("model explained 99 % of the variation").
Using the geometric short side the law is exact:

```
edge_m = 0.82*(L+S) + ditch_m + hedge_m + ε
residual std = 0.0575 m,  max |ε| = 0.107 m
```

so the analyst needs only the field dimensions; ditch/hedge are the
non-dimension terms.

### 2. The uniform signature (the actual hidden observation)

Distribution shape of the relevant quantities (n = 3869). A uniform
distribution has skew ≈ 0 and excess kurtosis ≈ −1.2:

| quantity | skew | excess kurtosis | KS vs uniform (sim. p) |
|---|---|---|---|
| **D = L + S** | **−0.03** | **−1.13** | 0.026 (p=0.02) |
| **edge_m** | **−0.02** | **−1.12** | 0.037 (p≈0.000) |
| ditch_m | −0.02 | −1.12 | 0.047 |
| hedge_m | −0.03 | −1.12 | 0.050 |
| long_side L | +0.21 | −0.81 | 0.210 |
| short side S | +0.58 | −0.16 | 0.282 |
| area_ha | +0.69 | −0.38 | 0.275 |

`D` (and thus the measured edge) is the only quantity that is
(unusually close to) flat. Crucially:

* `L` and `S` are individually **not** uniform;
* `corr(L, S) = +0.358` (positive!);
* nevertheless `D = L+S` is flat.

The generative construction is visible in the ratio
`f = L/D`: mean 0.722, std 0.093, skew +0.11, kurt −0.59 (bell shaped),
while `D` is flat. So `L = f·D`, `S = (1−f)·D` with `D ~ Uniform`
— a deliberate construction whose only purpose is to make the edge
length **uniformly distributed**.

A flat histogram of `edge_m` (the quantity Mariette measured) is the
observation; `uniform_evidence.png` shows it (left panel) next to D
(middle) and the individually non-uniform L, S (right).

## Reproduce

```bash
cd /home/kozak/mllm/MML/work/solutions/task_belgium
/home/kozak/mllm/py solve.py
```

Outputs:

* `uniform_evidence.png` — flat edge/D histograms vs non-flat L/S;
* `uniform_tests.csv` — skew / kurtosis / KS table;
* `answer.txt` — `SIGMOID_UNIFORM`.

## What was ruled out

* **field_id gaps / sample mask.** The 3869 present IDs out of the
  1…4945 range were rendered as bitmaps at every plausible width
  (43, 53, 73, 115 and every width 5–250), serpentine, transposed and
  rotated, row-normalised and column-normalised: the pattern is a smooth
  sampling-intensity gradient, no glyphs. Gap-run lengths (607 runs) are
  geometric and decode to nothing under 5/6/7/8-bit ASCII, Bacon, Morse,
  A1Z26 or base-4/5 schemes; no digit/prime/square/palindrome selection
  pattern (all |z| < 1.5).
* **Spatial word.** The 0.01° occupancy map is a realistic Flanders
  potato-field density (clusters agree with the real crop geography);
  population / density / rare-crop maps and thresholded subsets show no
  glyphs. Residuals have no spatial autocorrelation.
* **Text in numeric columns.** All decimal digit positions are uniform;
  parity / BCD / 4-bit / 5-bit / 6-bit / 7-bit / 8-bit decodings of every
  column, residual, category and ID sequence yield no dictionary words.
  The quote column is exactly `round(9.4·edge + 185)` (uniform rounding
  residual), and `crop` is a deterministic function of `crop_code`; both
  are redundant encodings, not carriers.
* **Model-structure word** (as in the Swiss task in this batch): a
  default decision tree on the declared features splits only on
  `ditch_m` / `long_side_m` / `area_ha`; feature-initial order gives no
  word.
* **Title word** "FRIENDSHIP": no 5-bit/8-bit/Bacon occurrence of it (or
  of other thematic words) in any categorical or binary sequence.

## Uncertainty / alternatives

The exact *word* the author intended cannot be confirmed inside the
data. The uniform construction is strong and deliberate, so `UNIFORM`
is the natural single word for "the word Mariette observed in her
measurements". Plausible runner-ups, if the intended observation is
instead about the sample-selection gradient (`p(id)` falls smoothly
≈ 1 → 0.4, and high-id surviving fields are larger), would be `BIAS`
or `SIGMOID` — but neither has a clean, exact data signature, whereas
the uniform `D`/`edge` construction is a non-generic, intentionally
built property. Confidence therefore: **medium**.

## Files

| file | purpose |
|---|---|
| `answer.txt` | answer candidate `SIGMOID_UNIFORM` |
| `solve.py` | reproducible solver (model + uniform tests + figure) |
| `uniform_evidence.png` | the key figure |
| `uniform_tests.csv` | skew/kurtosis/KS table |
| `map_all.png`, `grid_occupancy.png`, `density_sig*.png` | spatial null checks |
| `presence_*.png`, `bits_*.png`, `signals_w*.png` | ID-mask / binary null checks |
| `resid_*.png`, `pairs*.png` | residual/feature null checks |
