# task_france — "Point par point" (Easy, 100 pts, text flag)

**Answer candidate:** `SIGMOID_SEURAT`
**CONFIDENCE: low** — the word was **not** recovered from the data; `SEURAT` is
the thematic best guess (the story's pointillist founder).

> If the orchestrator wants a data-supported submission only, **do not submit
> this task yet**. If a guess must go in, `SIGMOID_SEURAT` is the candidate.

---

## 1. Task restated

Paris Airbnb listings (~28 k rows, 18 columns) are the dots of a pointillist
canvas. *Le Pointilliste* "slipped in some dots of their own"; "their colours
came from what every traveller weighs before booking a night". "Together they
form letters." Caption: **"Grey is pretty. It holds multiple secrets."**
Goal: find the word and read it "in the artist's order"; the scoring requires
code, a figure and a `labels.csv` that show the clustered word.

Reasonable intended pipeline (from the caption): map a numerical column
("colour" = price / money) to grey levels, plot the listings on Paris, and
cluster the dots that form letters.

## 2. Data facts (all reproduced by `solve.py`)

* `paris_listings.csv`: 28 108 rows × 18 columns, clean UTF-8, no extra files
  in the Drive folder (verified by the first solver pass).
* `listing_id` is **uniform on [1 000 358, 9 999 630]**: first-digit counts
  3 057/3 185/…/3 138, last-digit counts 2 728–2 856. IDs were re-generated and
  carry **no** information.
* Missing values: `beds` 933, `minimum_nights` 98, `bedrooms` 5 only (the
  hotel-room family). No missing `price`/`rating`.
* Coordinates: 48.816–48.902 N, 2.234–2.465 E (Paris intra-muros + the western
  suburb strip + Bois de Vincennes). Decimals follow exact 5-dp rounding
  (90.4 % 5 dp, 9.3 % 4 dp, 1.0 % 3 dp, 0.1 % 2 dp) — no hidden precision set.
* `price` cents are strongly structured: 12 921 integer, 3 735 halves, 1 899
  `x.67`, 1 251 `x.33`, plus quarters/fifths/tenths. Prices are rounded means of
  k integer nightly prices, with `minimum_nights` the dominant denominator for
  small k (mn = 2 → halves, 3 → thirds, 4 → quarters, 5 → fifths, 7 → sevenths).
  A separate **k/11** family exists (912 rows; 403 of the 888 `Room in hotel`
  rows; prices like 271.82, 353.64, 225.45).
* **Only clear generator fingerprint:** `calculated_host_listings_count == 225`
  occurs on exactly **7 rows**, and **6 of them have 3-decimal coordinates**
  (48.862/2.309, 48.853/2.336, 48.838/2.283, 48.848/2.287, 48.853/2.276,
  48.851/2.377), all `minimum_nights = 3`, `availability_365` 344–356,
  `host_is_superhost = False`. That is a real template signature, but 7 dots
  scattered across Paris cannot spell a word; it is evidence of the generator,
  not of the answer.

## 3. Negative results (what was ruled out)

Everything below was rendered and inspected (figures in `figs/`; larger
diagnostic set in `/tmp/opencode/france2/`):

| Probe | Result |
|---|---|
| All 28 108 points, equal-aspect, 1–2 km zoom tiles (12 tiles), 4 rotations/flips | ordinary Paris point cloud, no glyphs |
| Greyscale density (the literal "grey" reading), σ = 30–800 m, local-contrast ratio, blind de-trending | city shape + Poisson/building clumps only |
| Price / rating / reviews / availability / host-count / capacity maps (local median + local residual) | smooth fields, no letters |
| Every exact price, rating, availability, host-count, capacity value (≥6 rows): largest same-value spatial concentration | only `calculated_host_listings_count` (real multi-listing hosts) and `price = 54.39` (26 pts in 250 m, a building), no glyph |
| Narrow price (5/10/20/40 €) and rating (0.05/0.1/0.2) windows: local enrichment vs. base density | max 1–8 points per 300 m disk → noise |
| 50/100 quantile "shades of grey", price rounded to 1/2/5/10/20/25/50/100 € | no tight group |
| Missingness (beds / minimum_nights / bedrooms / any), hotel rooms, k/11 family, shared rooms, superhost, rare property types | spatially ordinary / hotel districts |
| Duplicate coordinates (1 039 locations, max multiplicity 16), mixed-label duplicates | building/host clusters, no letters |
| Sparse / isolated-point masks (`d31`, `d200` top 0.1–2 %), DBSCAN eps 30–100 m | city-edge points + one percolating city component |
| Lattice/phase scans: pitch 0.0001–0.01° and 10–200 m, 40×40 phase bins | no lattice (max bins consistent with rounding artefacts) |
| Nearest-neighbour distance histogram | spike at 0 m (duplicates), broad 8–16 m, **no glyph-pitch spike** |
| Nearest-neighbour direction histogram | 0°/90° peaks, but reproduced by a uniform null with 1e-5° rounding → artefact |
| Oriented matched filter (300×80 m, 150×40 m, 600×150 m), 12 orientations | street-like line network, no isolated letter strokes |
| KMeans (k = 6/8/12) on 85 scaled features, LOF(30), IsolationForest | clusters are geographic; outliers uniform |
| Row order (first/last 100–3 000 rows), ID-order, exact duplicates, acrostics in several orders | no message |
| Price/rating projections (lon–price, lat–price), aspect-1 degree space | no letters |

The point process itself is very **clumpy at building scale** (DBSCAN eps = 35 m
finds components of 3 732, 750, 294 … points), so a glyph made of added dots
would either (a) be in a low-density area and therefore obvious, or (b) be
distinguished by an attribute. Neither is observed in any of the ~40 probes.

## 4. Why the answer is only a guess

The word is not recoverable with the methods above. The caption most plausibly
means "price → grey → letters", and the only literal French wordplay that fits
the story is the double meaning of **argent** ("silver" = a pretty grey;
"money" = what a traveller weighs before booking). That is a possible
meta-answer, but it is not visible in the data either.

Candidate ranking (all thematic, no data proof):

1. **SEURAT** — Georges Seurat, the founder of pointillism named in the story;
   6 letters, fits a 6-glyph word. *Primary candidate.*
2. **POINTILLISM** — the art style; 11 letters, less likely as a dot-glyph word.
3. **ARGENT** — bilingual pun on "grey/silver/money" (see above).
4. **SIGNAC** — the second painter named in the story.
5. **POINTWISE / CLUSTER / SCATTER** — if the Antarctica slot flags are meant to
   be ML terms (TANH, UNIFORM are); `Point par point` ≈ pointwise/scatter.

## 5. Recommendation

* **Do not submit** unless the orchestrator accepts a thematic guess: the task
  is manually reviewed, and the required figure + `labels.csv` do not show any
  word, so a lucky word alone is unlikely to score.
* If a guess is required, use `SIGMOID_SEURAT`.
* A fresh angle worth trying (not done here): find the *original* public Paris
  Airbnb scrape and diff the coordinate set — the added dots would be the
  rows whose coordinates are absent from the real scrape. That is external
  data, so it was not used under the fair-play default.

## 6. Files

* `solve.py` — reproduces the audit, price-structure analysis, the 225
  fingerprint, the spatial/density/lattice probes and writes `figs/` +
  exploratory `labels.csv`.
* `labels.csv` — exploratory DBSCAN labels (`listing_id, dbscan_cluster`,
  eps = 200 m, min_samples = 5). **It does not show a word**; it is written only
  because the scoring asks for a labels file, not as evidence for the flag.
* `figs/01_all_points.png` … `figs/05_isolated.png` — diagnostic figures.
* `/tmp/opencode/france2/` — ~60 additional diagnostic figures from this pass.

## 7. Runtime

~90 min on 4 CPU cores; no external data, models or APIs used.
