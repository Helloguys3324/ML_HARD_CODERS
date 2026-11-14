# Solution Report: task_spain ("¡Feliz Navidad!")

## 1. Overview & Objective
- **Task ID:** `task_spain` (Spain).
- **Domain:** Tabular Data Analysis + Steganographic Bitmap Raster Reconstruction.
- **Output Format:** Text string `SIGMOID_<WORD>`.
- **Input Data:**
  - `office_price_guide.csv`: Catalog of 6 traditional Spanish souvenirs (`List_no` 1 to 6), unit of measurement, and fair price range in EUR:
    1. Jamon Iberico (kg): 45.00 – 90.00 EUR
    2. Churros (dozen): 4.00 – 8.00 EUR
    3. Turron de Jijona (bar): 6.00 – 14.00 EUR
    4. Paella kit (kit): 18.00 – 35.00 EUR
    5. Saffron (g): 7.00 – 15.00 EUR
    6. Barquillos (box): 5.00 – 10.00 EUR
  - `purchases.csv`: 577 purchase records across 21 airline pilots (`Pilot_no` 1 to 21), recording `Product`, `City`, `Quantity`, `Total_paid_eur`, and decoy features (shop rating, weather temperature, flight distance, etc.).
- **Rules & Mechanics:**
  - A receipt is **fair** iff the unit price ($\text{unit\_price} = \text{Total\_paid\_eur} / \text{Quantity}$) falls within $[\text{Fair\_min\_eur}, \text{Fair\_max\_eur}]$. Below is counterfeit ("fake"); above is a "tourist trap".
  - A board cell `(product, pilot)` is checked if that pilot purchased that souvenir at a fair price **at least once**.
- **Goal:** Classify purchases, construct a $6 \times 21$ coordinate grid (rows: products by `List_no`, columns: pilots by `Pilot_no`), and decode the secret word from the raster bitmap.

---

## 2. Key Insights & Methodology

### 2.1 Price Classification
- Computing $\text{unit\_price} = \text{Total\_paid\_eur} / \text{Quantity}$ partitions all 577 receipts into three cleanly separated clusters:
  - **Tourist Traps ($\text{unit\_price} > \text{Fair\_max\_eur}$):** 270 receipts (3.8× to 8.6× above normal).
  - **Fakes ($\text{unit\_price} < \text{Fair\_min\_eur}$):** 230 receipts (0.2× to 0.55× of minimum).
  - **Fair Purchases ($\text{Fair\_min} \le \text{unit\_price} \le \text{Fair\_max}$):** Exactly **77 receipts**.
- All auxiliary features (temperature, store rating, flight distance) are red herrings. `City` matches `Home_city` in 100% of rows.

### 2.2 Constructing the $6 \times 21$ Binary Matrix
- A binary matrix $M \in \{0, 1\}^{6 \times 21}$ is created:
  - Rows $r \in \{0, \dots, 5\}$: souvenirs ordered by `List_no` (1: Jamon, 2: Churros, 3: Turron, 4: Paella, 5: Saffron, 6: Barquillos).
  - Columns $c \in \{0, \dots, 20\}$: pilots by `Pilot_no` 1 to 21.
  - $M[r, c] = 1$ if pilot $c+1$ has $\ge 1$ fair purchase of souvenir $r+1$, else $0$.

### 2.3 Bitmap Glyph Decoding
- Columns **4, 8, 12, 16, and 18** are entirely zero, acting as 1-pixel letter spacers.
- Non-zero columns form clean 6-pixel high monospace raster glyphs:
  - **Cols 1–3 (width 3):** lowercase `c`
  - **Col 4 (width 1):** spacer
  - **Cols 5–7 (width 3):** uppercase `R`
  - **Col 8 (width 1):** spacer
  - **Cols 9–11 (width 3):** lowercase `u`
  - **Col 12 (width 1):** spacer
  - **Cols 13–15 (width 3):** uppercase `C`
  - **Col 16 (width 1):** spacer
  - **Col 17 (width 1):** narrow letter `i`
  - **Col 18 (width 1):** spacer
  - **Cols 19–21 (width 3):** uppercase `O`
- Column width breakdown: $3 + 1 + 3 + 1 + 3 + 1 + 3 + 1 + 1 + 1 + 3 = 21$ columns.
- Decoded word: **`CRUCIO`** $\to$ Flag: **`SIGMOID_CRUCIO`** (aligns with the Sigmoid open-source data augmentation library *Crucio* and the Harry Potter curse).

---

## 3. Implementation Details

1. **`solve.py` (Reproducible Pipeline):**
   - Merges `purchases.csv` with `office_price_guide.csv` on `Product`.
   - Filters receipts satisfying `Fair_min_eur <= unit_price <= Fair_max_eur`.
   - Populates a $6 \times 21$ NumPy array:
     ```python
     matrix = np.zeros((6, 21), dtype=int)
     for _, row in fair.iterrows():
         matrix[int(row["List_no"]) - 1, int(row["Pilot_no"]) - 1] = 1
     ```
   - Renders the raster grid to ASCII:
     ```
          123456789012345678901
        +---------------------+
     1  |....##......###...###|  (Jamon)
     2  |....#.#.....#...#.#.#|  (Churros)
     3  |###.#.#.#.#.#.....#.#|  (Turron)
     4  |#...##..#.#.#...#.#.#|  (Paella)
     5  |#...#.#.#.#.#...#.#.#|  (Saffron)
     6  |###.#.#.###.###.#.###|  (Barquillos)
     ```
   - Submits `SIGMOID_CRUCIO` to the platform API.

2. **`solution.md` (Robustness Verification):**
   - Alternative aggregation thresholds ("all fair" $\to 0$ px; "majority fair" $\to 6$ dots; "$\ge 2$ fair" $\to 20$ scattered dots) destroy glyph continuity; the text appears uniquely at "$\ge 1$ fair purchase".
   - Any reordering of rows (alphabetical, price-sorted) or columns yields unreadable noise.

---

## 4. Results & Artifacts
- **Final Flag:** `SIGMOID_CRUCIO`
- **Confidence:** High (unambiguous monospace glyph rendering with exact column delimiters).
- **Key Artifacts (`task_spain/`):**
  - `answer.txt`: Contains `SIGMOID_CRUCIO`.
  - `solve.py`: End-to-end data processing, ASCII rasterization, and submission script.
  - `solution.md`: Mathematical proofs, ASCII art, and robustness analysis.
