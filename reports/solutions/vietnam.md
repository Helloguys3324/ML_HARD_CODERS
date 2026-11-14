# Solution Report: task_vietnam ("That isn't my shoe!")

## 1. Overview & Objective
- **Task ID:** `task_vietnam` (Vietnam).
- **Track:** Computer Vision (`cv`).
- **Difficulty:** Medium (200 points).
- **Submission Type:** Serialized model file `main.pt` (TorchScript / PyTorch checkpoint), strictly $\le 10.0$ MB (10,485,760 bytes).
- **Premise:**
  - Vietnam is a primary manufacturing hub for athletic footwear (Nike, Adidas, Converse).
  - An automated assembly-line scanner categorizes shoes into three known classes: `adidas`, `converse`, `nike`.
  - Non-target brands (Puma, Reebok, Vans, Asics), counterfeits, and non-athletic shoes also enter the conveyor. Closed-set classifiers erroneously force unseen brands into one of the three known labels.
  - The scanner must perform open-set recognition: correctly classify authentic known brands while rejecting unknown or anomalous shoes into an **`unknown`** class.
- **Input Data:**
  - `train-input/` and `train-output.csv`: 735 real sneaker images ($256 \times 256$ pixels), labeled strictly into `adidas`, `converse`, and `nike`.
  - The test set is retained by the grading platform and contains a mixture of known models and unseen brands.
- **Evaluation Criteria (Dual Strict Thresholds):**
  - **Known Accuracy** $\ge 0.85$ (accuracy on known brand images).
  - **Rejection Rate** $\ge 0.70$ (fraction of unseen brand images routed to `unknown`).
  - Both thresholds must be satisfied concurrently to pass and trigger flag release.

---

## 2. Key Insights & Methodology

### 2.1 Problem Formulation: Open-Set Recognition (OSR / OOD Detection)
- Standard softmax classifiers suffer from extreme overconfidence on out-of-distribution (OOD) samples.
- The $\le 10.0$ MB model constraint requires lightweight, compact architectures (custom ResNet micro-architectures or MobileNetV3) compiled and serialized via TorchScript.

### 2.2 Classical Feature Baseline (`classical.py`)
- Extracted HOG descriptors ($16 \times 16$ cell, $2 \times 2$ block), spatial RGB color histograms ($4 \times 4$ zones, 8 bins/channel), and Local Binary Patterns (LBP).
- Evaluated `SVC(RBF)`, `LinearSVC`, and `LogisticRegression` via 5-fold cross-validation.
- Best cross-validated accuracy capped at **71.3%**, falling far short of the 0.85 threshold and confirming the necessity of deep convolutional features.

### 2.3 Deep Convolutional Architecture (`ShoeNet` / `MobileNetV3`)
- **Backbone:** 2-stage stem (stride-2 $3 \times 3$ convolutions) with 4 residual stages of `BasicBlock` modules with channel progression ($w \to 2w \to 4w \to 8w$).
- **Generalized Mean (GeM) Pooling:**
  $$\text{GeM}(x) = \left( \frac{1}{|R|} \sum_{u \in R} x_u^p \right)^{1/p}$$
  Learnable parameter $p$ adaptively interpolates between average and max pooling, sharpening focus on logos and stitching details.
- **Metric Projection Neck:** 256-dimensional linear layer + BatchNorm1d + L2 normalization for distance-based rejection.

### 2.4 OOD Detection & Synthetic Outlier Exposure
Tested OOD scoring formulations:
- **Maximum Softmax Probability (MSP):** Thresholding $\max_c P(y=c|x) < \theta$.
- **Logit Energy Score:** $E(x) = -\log \sum_{c=1}^C e^{f_c(x)}$, offering greater separation than uncalibrated probabilities.
- **Cosine Prototype Similarity:** Distance to training class centroids $\mu_c = \frac{1}{|S_c|} \sum_{i \in S_c} e_i$.
- **Synthetic Jigsaw Outlier Exposure (`train.py`):**
  Images are divided into a $4 \times 4$ grid of $64 \times 64$ patches and randomly shuffled spatially. The model is trained to minimize cross-entropy on genuine images while maximizing prediction entropy (uniform distribution) on jigsaw collages:
  $$\mathcal{L} = \mathcal{L}_{\text{CE}}(\text{logits}_{\text{id}}, y) + 0.3 \cdot \mathcal{L}_{\text{entropy}}(\text{logits}_{\text{jigsaw}})$$
  This penalizes overconfident predictions on broken visual structures.

### 2.5 Leave-One-Brand-Out (LOO) Validation Protocol
- To calibrate rejection thresholds without external OOD data, each brand is held out in turn as pseudo-unknown.
- The rejection cutoff is tuned on the remaining two brands at a False Rejection Rate $\text{FRR} \le 0.10$ (preserving $\ge 0.90$ known accuracy), and verified against the held-out brand to ensure $\ge 0.70$ rejection.

---

## 3. Implementation Details

1. **`classical.py` (Classical Vision Baseline):**
   - Implements HOG, spatial color histograms, and LBP feature extractors.
   - 5-fold `StratifiedKFold` evaluation reporting 71.3% accuracy ceiling.

2. **`train_cnn.py` (Local PyTorch Training):**
   - Custom `ShoeDataset` with aggressive augmentations: `RandomResizedCrop(0.5, 1.0)`, `RandomHorizontalFlip`, `RandomRotation(15)`, `ColorJitter`, and `RandomErasing(p=0.2)`.
   - Regularization: Exponential Moving Average (EMA, decay=0.999), Mixup ($\alpha=0.2$), Label Smoothing ($\epsilon=0.1$), and AdamW with cosine annealing.

3. **`train_fast.py` (GPU-Accelerated In-Memory Pipeline):**
   - Preloads all 735 images directly to GPU VRAM as `uint8 [N, 3, 256, 256]`.
   - Computes spatial transforms and color augmentations entirely on GPU tensors via `F.affine_grid` and `F.grid_sample`.

4. **`train_modal.py` (Modal Cloud Orchestration):**
   - Supports CLI modes: `--mode cv` (5-fold benchmark), `--mode loo` (leave-one-brand-out threshold calibration), and `--mode final` (full-data training).
   - Utilizes mixed-precision (`torch.autocast`, bfloat16/float16 + `GradScaler`).

5. **`tasks/vn_vietnam/work/train.py` (TorchScript Artifact Generation):**
   - Fine-tunes lightweight MobileNetV3 with Jigsaw entropy regularization.
   - Exports the model via `torch.jit.script` into `main.pt`.
   - Resulting artifact size: **~9.2 MB**, strictly within the 10.0 MB grading limit.

---

## 4. Results & Artifacts
- **Status:** Prepared for platform evaluation (200 points).
- **Submission Artifact:** `main.pt` (TorchScript model, ~9.2 MB).
- **Target Performance Metrics:**
  - **Known Accuracy:** $\ge 0.85$ (achieves $> 0.90$ in cross-validation).
  - **Rejection Rate:** $\ge 0.70$ (achieves $> 0.75$ on held-out brand validation).
- **Key Artifacts (`task_vietnam/`):**
  - `classical.py`: Reproducible HOG + LBP baseline.
  - `train_cnn.py`: `ShoeNet` architecture with GeM pooling and EMA.
  - `train_fast.py`: Fast GPU-cached training script.
  - `train_modal.py`: Modal cloud trainer supporting CV, LOO calibration, and final export.
  - `main.pt`: Compiled TorchScript model file.
