# task_spain — ¡Feliz Navidad!

**Answer:** `SIGMOID_CRUCIO`
**Confidence:** **high**

## TL;DR

Applying the office rule (unit price inside the guide's fair range) to the 577
receipts and pivoting the surviving purchases as **rows = 6 souvenirs (List_no
order) × columns = 21 pilots** produces a 6×21 pixel image. The blank columns
are exactly the letter spacers, and the glyphs read:

```
c   R   u   C   i   O
```

→ **CRUCIO** → flag `SIGMOID_CRUCIO`.

## Method (implemented in `solve.py`)

1. Load `office_price_guide.csv` (6 products, fair EUR range per unit) and
   `purchases.csv` (577 receipts, 14 columns).
2. Compute the **unit price** `Total_paid_eur / Quantity`; `Quantity` is already
   expressed in the guide's `Unit` (kg / dozen / bar / kit / g / box).
3. Classify every receipt:
   * `unit_price < Fair_min_eur` → **fake**
   * `unit_price > Fair_max_eur` → **tourist trap**
   * otherwise → **fair** (inclusive range).
4. Build the board exactly as described: a cell `(product, pilot)` is checked
   iff that pilot bought that product at a fair price **at least once**
   ("HR checks which pilot brought a given souvenir for a fair price").
   Rows use the office list order `List_no = 1..6`; columns are `Pilot_no = 1..21`.
5. Render the 6×21 binary grid with square pixels; split the columns on the
   all-zero columns (letter spacers) and read the glyphs.

Run:

```bash
/home/kozak/mllm/py solve.py \
  /home/kozak/mllm/MML/work/tasks/task_spain/data \
  /home/kozak/mllm/MML/work/solutions/task_spain
```

Output: `answer.txt`, `board_solution.png` (figure with the word visible),
ASCII board printed to stdout.

## Evidence

### Receipt classification (577 rows)

| verdict | count |
|---|---|
| tourist trap (`> max`) | 270 |
| fake (`< min`) | 230 |
| **fair** | **77** |

The three price clusters are cleanly separated (fake ≈ 0.2–0.55 × fair-min;
fair ≈ 1.1–2.3 × fair-min; trap ≈ 3.8–8.6 × fair-min), so classification is
not sensitive to rounding. Using a strict interior range, or rounding unit
prices to cents first, produces the **identical** board.

`City` always equals the product's `Home_city` (100 %), so the "buy it in its
home city" story adds no extra filter.

### The reconstructed board

`#` = checked (≥1 fair purchase for that pilot × product):

```
      123456789012345678901
Jamon    ....##......###...###
Churros  ....#.#.....#...#.#.#
Turron   ###.#.#.#.#.#.....#.#
Paella   #...##..#.#.#...#.#.#
Saffron  #...#.#.#.#.#...#.#.#
Barquill ###.#.#.###.###.#.###
```

All 126 pilot×product cells contain at least 3 receipts, so a blank cell is a
genuine absence of a fair purchase, not missing data.

### Reading the word

All-zero columns: **4, 8, 12, 16, 18** — exactly the spacers between glyphs.
The column groups and their bitmaps (top row = Jamon, bottom = Barquillos):

| # | pilot cols | bitmap (6 rows × w) | letter |
|---|---|---|---|
| 1 | 1–3 | `...` / `...` / `###` / `#..` / `#..` / `###` | **C** (lower-case c) |
| 2 | 5–7 | `##.` / `#.#` / `#.#` / `##.` / `#.#` / `#.#` | **R** |
| 3 | 9–11 | `...` / `...` / `#.#` / `#.#` / `#.#` / `###` | **U** (lower-case u) |
| 4 | 13–15 | `###` / `#..` / `#..` / `#..` / `#..` / `###` | **C** |
| 5 | 17 | `.#.###` (dot + stem, 1 px wide) | **I** (lower-case i) |
| 6 | 19–21 | `###` / `#.#` / `#.#` / `#.#` / `#.#` / `###` | **O** |

The layout is self-consistent: each letter is 3 columns wide except the
1-column `i`; with 1-column spacers the 21 pilots decompose exactly as
3+1+3+1+3+1+3+1+1+1+3 = **21**. Uppercasing the lowercase-shaped glyphs gives
**CRUCIO** (the task explicitly warns that some glyphs may be lowercase-shaped
and should be submitted uppercase).

Visual check: `board_solution.png` — the left panel shows the board as
`c R u C i O` when rendered with square pixels.

### Alternatives ruled out

* **Wrong aggregation**: "all receipts fair" → 0 pixels; "majority fair" →
  only 6 scattered pixels; "at least 2 fair" → 20 scattered pixels. Only
  "≥ 1 fair purchase" yields a coherent word.
* **Wrong rule**: the fake-only and trap-only boards are almost fully dense
  (112 / 121 pixels) and show no glyphs; the complement of the fair board is
  noise too.
* **Wrong orientation / row order**: transposing, rotating by 90°/180°, or
  mirroring destroys the letters. Alphabetical / by-home-city /
  reverse-List_no / by-count / by-price row orders all produce gibberish;
  only the office `List_no` order (Jamon, Churros, Turron, Paella, Saffron,
  Barquillos) gives clean glyphs. Likewise only the natural `Pilot_no` column
  order spells the word.
* **Numeric-column red herrings** (`Flight_km`, `Temperature_c`,
  `Luggage_kg`, `Shop_rating`, `Payment`, dates, shop types) do not affect the
  board; they are decoys.

### Plausibility of the word

`CRUCIO` is a real word (Latin "I torture"; the Harry Potter Cruciatus curse)
and, notably, the name of a **Sigmoid** open-source library
("Crucio — torture your unbalanced data"), so it fits an auto-generated
Sigmoid task word list. Independently of the theme, the pixel image is
unambiguous: every glyph matches the board's consistent 3-wide pixel font.

## Files

* `solve.py` — reproducible solution (pandas/numpy/matplotlib).
* `board_solution.png` — figure: left = reconstructed board showing the word,
  right = receipt classification (fake / fair / trap).
* `answer.txt` — `SIGMOID_CRUCIO`.
