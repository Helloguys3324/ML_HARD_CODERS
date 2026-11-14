# Solution Report: task_kazakhstan ("Stamp of Disapproval")

## 1. Overview & Objective
- **Task ID:** `task_kazakhstan` (Kazakhstan).
- **Task Type:** Image Forensics / Tampering Detection; text response (`answer.txt`).
- **Dataset:**
  - 5 stock-book albums of 26 pages each (130 JPEG images at $1600 \times 1200$).
  - 30 calibration exhibits — postage stamp cutouts exhibiting representative editing artifacts, each paired with a ground-truth binary PNG tamper mask.
- **Mechanics & Objective:**
  - Each page displays 3 postage stamps from Kazakhstan.
  - On exactly one stamp per page, a small rectangular region has been digitally modified (e.g., denomination/year digit replaced, element cloned, or text erased).
  - Plotting the center coordinates of tampered regions across all 26 pages of an album reveals the shape of one letter glyph.
  - The 5 albums form a 5-letter word: `SIGMOID_<5_LETTER_WORD>`.
  - External stamp catalogs are uninformative due to multi-denomination definitive series; detection must rely purely on pixel-level forensic analysis.

---

## 2. Key Insights & Methodology

1. **Quantization Table Invariance:**
   - Both album pages and calibration exhibits share identical native JPEG quantization tables (`[1,1,1,1,1,2,3,4,…]`, native quality $q \approx 97$), confirming they originate from the same compression pipeline.

2. **Tampering Geometry & Localization:**
   - Altered regions are strictly axis-aligned bounding boxes with an average size of $63 \times 41$ pixels (range: $25 \times 21$ to $103 \times 55$).
   - Centers cluster strongly around stamp corners (lower-left: 9, upper-right: 9, upper-left: 5, lower-right: 2, mid-right: 3), where denominations, release years, "Kazakhstan / Qazaqstan", and artist signatures are positioned.

3. **Three Tampering Signatures:**
   - **Flat Fill (Text Erasure):** Complete suppression of high-frequency components. Constant RGB fill (std $\approx 0.8 \dots 3.7$), which at $q \approx 97$ yields byte-identical values across $60 \times 40$ px (impossible on real scanned paper fibers). Manifests as a sharp black box in high-pass filtered images (Exhibits 1, 4, 10, 16, 28).
   - **Ghost Removal:** Semi-transparent stroke blending that leaves faint remnant outlines (Exhibits 11, 14: erased artist names).
   - **Digit Replacement / Splicing:** Inpainted box with redrawn or cloned digits (Exhibits 2, 7, 9, 12, 13, 17, 25). Typically exhibits a 1-pixel boundary seam and ghost contours from the original glyph.

4. **Error-Level Analysis (ELA):**
   - At native $q = 97$, ELA displays a characteristic bipolar response: significantly below average for flat fills and above average for redrawn or spliced digits.

---

## 3. Implementation Details

Implemented in `solve.py` using NumPy and OpenCV:

1. **Stamp Page Segmentation (`stamps_on`):**
   - Measures pixel distance from album background `BG = [38, 36, 34]`.
   - Applies thresholding (`d > 30`) and morphological closing (`cv2.morphologyEx` with a $9 \times 9$ kernel).
   - Extracts bounding boxes via `cv2.connectedComponentsWithStats` ($w > 100, h > 100$). Successfully isolates 3 stamps on 126 of 130 pages (remaining 4 contain souvenir sheets/attached coupons) and sorts them in reading order.

2. **Forensic Filtering:**
   - `highpass(gray, sigma=1.2)`: Gaussian difference filter accentuating high-frequency paper grain and stroke boundaries.
   - `local_std(gray, k=7)`: Computes local standard deviation within a $7 \times 7$ window via two `cv2.boxFilter` passes.
   - `ela(page, q=97)`: Computes recompression residual maps at native JPEG quality $q = 97$.

3. **Empirical Detections:**
   - `page_1759` (Album 1), stamp 1, denomination "202": The trailing digit "2" is an exact pixel clone of the leading "2" ($|\Delta| = 4.8$), with underlying ghost contours visible.
   - `page_7501` (Album 1), stamp 1: Monochromatic $71 \times 32$ px box with an 8.3 luminance delta against the surrounding ring (matching Exhibit 1).

4. **Decoding Limitation:**
   - The zero-std flat-fill detector achieves 100% precision but low recall (1 of 26 pages in Album 1). Broader detectors (ELA, local flatness) trigger false positives on saturated color blocks, gold foil embossing, and stamp perforations. Because glyph rendering requires near-complete 26-point coverage, resulting scatter plots contained too much noise for unambiguous OCR.

---

## 4. Results & Artifacts
- **Output Files:**
  - `answer.txt`: Formulates the baseline thematic flag hypothesis `SIGMOID_STAMP`.
  - `exhibit_stats.csv`: Statistical profiles across the 30 ground-truth exhibits.
  - `boxes_candidates.csv`: Bounding box coordinates for detected anomaly candidates.
  - `out/`: Page-level dot plots and forensic residual maps.
- **Confidence Assessment:**
  - **Confidence: LOW.** `SIGMOID_STAMP` is a reasoned domain hypothesis ("Stamp of Disapproval", 5 albums = 5 letters) rather than a complete geometric reconstruction from all 130 pages. Full forensic tools and metrics are preserved in `solve.py` and `solution.md`.
