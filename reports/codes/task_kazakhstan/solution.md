# Stamp of Disapproval — solution notes (`task_kazakhstan`)

**Answer candidate:** `SIGMOID_STAMP` (see confidence note at the bottom)
**Confidence:** low — the flag was **not** independently recovered; the candidate is a
thematic best guess. The forensic method and the failure analysis are documented
in full so the next pass can continue from the strongest leads.

---

## 1. Task, as specified

* Dataset: 5 stock-book albums × 26 pages (1600×1200 JPEG) + 30 "exhibits"
  (single stamps altered in the same ways, each with a binary mask PNG).
* On every album page exactly **one of the three stamps** has one small
  rectangular region altered: *digit changed*, *element cloned* or
  *inscription removed*.
* Marking the altered region of every page of one album on a single sheet must
  spell a **letter**; five albums ⇒ five letters ⇒ `SIGMOID_<WORD>`.
* Comparing stamps against external catalogues is explicitly not detection;
  this solution only ever uses the supplied pixels.

## 2. Data audit (observed)

* 130 album pages, 30 exhibits + 30 masks, `baseline_solution.py`,
  `requirements.txt`; nothing else is in the Drive folder.
* JPEG quantisation tables of **pages and exhibits are identical**
  (`[1,1,1,1,1,2,3,4,…]`, native quality ≈ 97). Exhibits are therefore crops
  from the same generation pipeline as the pages (they even include the dark
  stock-book background).
* Mask statistics: axis-aligned **rectangles**, fill ≥ 0.84, median size
  **63 × 41 px** (range 25×21 … 103×55).
* Mask centres relative to the stamp frame cluster hard at the stamp
  **corners** (bottom-left 9×, bottom-right 2×, top-right 9×, top-left 5×,
  mid-right 3×). This matches where denominations, years, country names and
  designer names sit on Kazakhstan stamps.
* Stamp segmentation (`stamps_on`) gives 3 components on 126/130 pages; the
  remaining four contain se-tenant pairs/strips (one item is a two-stamp
  block, one portrait stamp splits into two components).

## 3. What an alteration looks like (calibrated on the 30 exhibits)

Three families are visible:

1. **Flat fill / “inscription removed”** — the high-pass content of the patch
   collapses. For exhibits 1, 4, 10, 16, 28 the patch is an
   **exactly constant RGB colour** (std ≈ 0.8–3.7; the largest exactly-equal
   pixel component coincides with the mask). At JPEG q≈97 a single-colour
   fill has zero AC coefficients, so it decodes back to byte-identical values —
   something scanned paper never does over a 60×40 px area.
   Evidence: `figures/exhibit_highpass_flatfill.jpg` (patches appear as a
   perfectly black rectangle with a sharp boundary in the high-pass view).
2. **Ghost removal** — the inscription is alpha-blended toward the paper
   colour, leaving low-contrast “ghost” strokes inside a rectangle
   (e.g. exhibit 14: the middle of “Д. МУХАМЕДЖАНОВ” is washed out but still
   visible; exhibit 11: faded “Т. Айбакуров”).
3. **Digit / element replacement** — a pasted patch with a flat background
   plus a re-rendered digit or a duplicated element (exhibits 2, 7, 9, 12,
   13, 17, 25). The patch background has essentially no JPEG/texture noise
   while the host stamp has, and the mask boundary often carries a 1-px seam.
   Evidence: `figures/exhibit_highpass_digits.jpg`.

Also measured: ELA (error-level analysis) at the *native* quality 97 is
**bipolar** for edited patches — much lower than average for flat fills and
much higher for re-rendered digits (rank percentile ≤0.15 or ≥0.75 for about
two thirds of the exhibits). At the baseline's quality 90 the whole stamp
lights up instead, which is why the provided baseline only finds stamps.

## 4. Detectors tried on the album pages

| Detector | Result |
|---|---|
| ELA at q97, 9×9 box, max inside stamps (baseline idea, native quality) | Locates saturated red/blue lettering (e.g. page_1759 “250”), not consistently the edit. Dot plots: `figures/ela_peak_dots_album1.png` — no letter. |
| ELA residual after regressing out local texture slope | Reduces the colour-saturation bias, still too noisy. |
| Flat-fill detector (7×7 local std threshold + rectangular opening + colour step to ring) | Precision too low on busy stamps (shiny gold foil, white paper). Dot plots: `figures/flatfill_dots_album1.png` — no letter. |
| Exactly-constant-colour rectangle detector | **High precision, low recall.** Reproduces exhibit 1’s fill exactly; on album_1 it fires on only **1/26 pages** (page_7501, 71×32 fill at page (567,452)-ish). |
| SIFT / block self-matching for clone-move | Swamped by genuine repetitions (perforations, ornaments, repeated digits). No stable translated-keypoint cluster at the edit boxes. |
| Per-stamp JPEG grid-phase / blockiness | Ratio ≈1.0 on all stamps: pages are single JPEG encodes, so no compression-history outlier. |
| Compare duplicate stamp designs across pages | Only series variants (Gladiolus 10/20/50, etc.) — different denominations, not pristine copies. Confirms the task’s warning that catalogue-style comparison is useless. |

## 5. Concrete page-level findings (observed, not inferred)

* **page_1759 (album_1), stamp 1 “202”.** The last “2” is a near-exact
  duplicate of the first “2” (best alignment at (0,0), mean |Δ| = 4.8 grey
  levels over the glyph) while the middle “0” is a different glyph, and the
  digit carries faint extra strokes on its lower-left and upper-left
  (`figures/page1759_denom202_ghost.jpg`; high-pass view shows the ghost).
  This is the classic “digit changed” signature from exhibits 2/7/9/12: a
  digit is replaced by a copy of another digit in the same number, leaving
  the old glyph’s rim visible. “202” is also a denomination that raises the
  storyline’s “value that never existed” flag.
* **page_1759, stamp 0 “250”.** Strongest ELA outlier of the page
  (13.7 vs 8.9 / 6.5 for the other stamps, `figures/page1759_ELA_q97.jpg`),
  localised on the red “250” script. May be a re-rendered denomination
  (the “5” has an unusual angular top and a faint double rim) or simply the
  most saturated text on the page; unresolved.
* **page_7501 (album_1), stamp 1.** Contains a 71×32 px **exactly constant
  RGB** rectangle with a colour step of 8.3 grey levels to its ring — by
  itself a very strong “painted fill” indicator, matching exhibit 1’s
  signature.
* Stamp-corner crops for inspection: `figures/page1759_corners.jpg`.

## 6. Why the letter was not recovered

The intended dot for each page is the **centre of the small altered
rectangle** (the baseline draws exactly such centres and the brief calls the
marks “regions … on a single sheet”). The authors’ own baseline shows that
this geometry is unforgiving: a detector that does not hit the *right stamp*
and the *right corner* on nearly every page turns the 26-point letter into
noise.

My strongest detectors are high-precision but low-recall (exact-colour fill
fires on 1/26 pages of album_1). The high-recall cues (ELA, flatness) have
too many false positives on saturated lettering, gold foil, white paper and
perforations. Without at least ~10–15 correct points per album, no letter
could be read from the plots; all five candidate dot plots are posted in
`figures/` and none forms a recognisable glyph.

## 7. Recommended next steps (highest expected value)

1. **Ghost detector, text-aware.** Build a text-stroke mask (dark-on-light
   and light-on-dark), then look for rectangular regions where the same
   strokes appear with 20–50 % of the contrast of the surrounding text —
   exhibits 11/14 show this exactly. This should recover “inscription
   removed” pages and is cheap.
2. **Digit-pair clone test.** For every number in a stamp, check every digit
   pair for pixel identity after integer alignment (exhibits 2/7/9/12 all
   show a copied digit with the old rim ghosting). Pair with the
   “denomination never existed” prior from the storyline.
3. **Exact-colour fill + seam detector** for the remaining “removed” pages.
4. Only then re-plot the 26 centres per album (say 1 %–99 % robust centroid of
   the detection) and read the letters; cross-check with the 5-letter word.

## 8. Answer and confidence

`answer.txt` contains `SIGMOID_STAMP`.

**CONFIDENCE: low.** The word is a thematic guess (“Stamp of Disapproval”,
five albums ⇒ five letters), not the output of a completed letter read-out.
The detection machinery and all intermediate evidence are nevertheless real
and reproducible (`solve.py`, `figures/`), so if the orchestrator has another
90-minute slot the ghost/clone detectors above are the shortest path to the
true five letters.

## 9. Reproducibility

```
cd work/solutions/task_kazakhstan
/home/kozak/mllm/py solve.py --data ../../tasks/task_kazakhstan/data --out out --album 1
```

Verified output: `30 exhibits analysed`, `album 1 ela_residual: 26 dots`,
`album 1 flatfill: 1 dot`; artefacts written to `out/` and copied here as
`exhibit_stats.csv`, `boxes_candidates.csv`, `figures/dots_album1_*.png`.

Runtime of the full analysis in this session: ≈ 2 h wall clock (data fetch
≈ 2 min, page/JPEG sweeps dominate). No external datasets, models or APIs
were used; every byte of task data was treated as passive data.
