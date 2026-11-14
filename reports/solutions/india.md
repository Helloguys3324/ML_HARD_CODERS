# Solution Report: task_india ("Twins Separated at Birth")

## 1. Problem Overview
- **Task & Track:** Duplicate Image Pair Detection (100 pts).
- **Input Data:** Collection of scrambled photographs of Indian monuments. The training set (`train/`) contains 4,057 JPEG images (`img_000001.jpg` – `img_004057.jpg`).
- **Ground Truth:** `train-output.csv` with 1,399 true duplicate pairs across 1,119 connected components (979 pairs, 140 triplets; 2,378 images in duplicates, 1,679 singletons).
- **Target Metric:** Pairwise F1-score with passing threshold **$F1 \ge 0.88$**.
- **Submission Format:** Standalone script `main.py` producing `output.csv` with columns `image_a,image_b` (unordered pairs, $a \neq b$).

## 2. Key Insights & Dataset Invariants
SIFT and RANSAC alignment revealed the synthetic distortion pipeline:
1. Cropping a central sub-window with zoom factor $0.66 \dots 1.1$ and center drift $\le \pm 7\%$.
2. Resizing to new dimensions and JPEG re-encoding (with occasional Gaussian blur).
- **Aspect Ratio Invariant:** Across all 1,399 labeled pairs, $\min(AR_a, AR_b) / \max(AR_a, AR_b) \ge 0.989$ (0 violations).
- **Global pHash Limitation:** Standard whole-image pHash captures only $\approx 55\%$ of pairs at Hamming distance $\le 8$ due to cropping and repeated monument structural symmetry.
- **Hard Negatives:** Alternate photos of the same monument taken minutes apart yield Normalized Cross-Correlation (NCC) up to 0.99 and $>200$ RANSAC inliers, necessitating strict multi-condition verification.

## 3. Two-Stage Detection Architecture (`main.py`)
Implemented entirely with standard Python, NumPy, and OpenCV (no deep neural networks or external API calls):

1. **Candidate Retrieval (Multi-Zoom pHash + Aspect Ratio Filter):**
   - 64-bit DCT perceptual hash computed across 8 central crop scale factors:
     $$\text{FRACS} = [1.0, 0.9, 0.84, 0.78, 0.72, 0.66, 0.60, 0.54]$$
   - Matches full image $A$ against all 8 zoom hashes of $B$ (and vice versa) using vectorized bitwise popcounts (`_popcount`).
   - Pairs with directional Hamming distance $\le 20$ and $AR \ge 0.97$ are retained. Yields 81,622 candidate pairs for 4,057 images ($\approx 97\%$ true duplicate recall).
2. **Geometric Verification (ORB + RANSAC Affine Partial + Overlap NCC):**
   - Rescales images to max dimension 512, extracts features via `cv2.ORB_create(nfeatures=350, fastThreshold=10)`.
   - Matches descriptors using `cv2.BFMatcher(cv2.NORM_HAMMING)` with Lowe's ratio test threshold 0.75.
   - Computes partial affine similarity (rotation, translation, isotropic scale) using `cv2.estimateAffinePartial2D` with RANSAC (reprojection error 3.0 px, 2,000 iterations).
   - Warps $A$ onto $B$, computes overlap mask ($\ge 200\,\text{px}$, area ratio $\ge 12\%$), and evaluates Normalized Cross-Correlation (NCC).
   - **Acceptance Rule:**
     - Primary: inliers $\ge 40$, inlier ratio $\ge 0.60$, $\text{NCC} \ge 0.70$, and scale $s \in [0.35, 1.45]$.
     - Strong-hash fallback (for heavily blurred images): Hamming distance $\le 8$, $AR \ge 0.985$, and inliers $\ge 20$.

## 4. Results & Artifacts
- **Full Training Set Metrics (4,057 images, 1,399 ground truth pairs):**
  - True Positives (TP): **1,359**
  - False Positives (FP): **47** (predominantly identical monuments photographed seconds apart)
  - False Negatives (FN): **40** (extreme crop or severe blur)
  - Precision: **0.967**
  - Recall: **0.971**
  - **F1 Score: 0.969** (comfortably exceeding threshold **0.88**).
- **Isolated Smoke Test (119 images):** 33 pairs detected, FP = 0, F1 = 0.971.
- **Runtime:** $\approx 4$ minutes on 4 CPU cores (pHash $\approx 74\,\text{s}$, candidate filter $\approx 5\,\text{s}$, ORB $\approx 69\,\text{s}$, verification $\approx 279\,\text{s}$).
- **Files:** `main.py`, `output.csv`, `sweep.py`, `solve.py`.
