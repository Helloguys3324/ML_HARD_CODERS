# Solution Report: task_ecuador ("Ciclopaseo Mirage")

## 1. Overview & Objective
- **Task ID:** `task_ecuador` (Ecuador).
- **Difficulty & Score:** Medium, 200 / 200 points.
- **Domain:** Adversarial Machine Learning, Computer Vision, White-Box Evasion Attacks.
- **Input Data:** `train-input.png` (224×224 RGB image, ImageNet class 671: `mountain bike`).
- **Target Classifier:** MobileNetV3-Large (`IMAGENET1K_V2`), checkpoint `mobilenet_v3_large-5c1a4163.pth`.
- **Target Objective:** Manipulate the input into `test-output.png` such that the model predicts class 665 (`moped`) with softmax confidence **$\ge 0.99$**.
- **Adversarial Constraint ($L_\infty$-ball):** Perturbations per color channel are strictly constrained to at most $\pm 1$ on the integer $[0, 255]$ pixel scale ($\epsilon = 1 / 255 \approx 0.00392$).
- **Runtime Budget:** Complete execution in under 60.0 seconds.

---

## 2. Key Insights & Attack Methodology

### 2.1 White-Box Optimization Dynamics
Access to exact model parameters and input preprocessing transformations allowed direct end-to-end backpropagation of loss gradients with respect to the input tensor, bypassing black-box transfer limitations.

### 2.2 Semantic Proximity of Target Classes
`mountain bike` (671) and `moped` (665) share fundamental morphological visual primitives (spoke wheels, handlebars, dual-chassis frame). The decision boundary between these two classes lies close in representation space, enabling high-confidence class flipping even under an extremely constrained $\pm 1$ pixel budget.

### 2.3 Projected Momentum Iterative Descent (MI-FGSM / PGD)
- **Targeted Cross-Entropy Objective:**
  $$\mathcal{L} = \operatorname{CrossEntropy}(\operatorname{logits}(x_0 + \delta), [665])$$
- **Velocity Accumulation with Decay ($\mu = 0.9$):**
  $$g_{t+1} = \mu \cdot g_t + \nabla_{\delta} \mathcal{L}$$
- **Sign Update with $L_\infty$ Box Projection:**
  $$\delta_{t+1} = \operatorname{clamp}\left(\delta_t - \frac{1}{255} \operatorname{sign}(g_{t+1}), -\frac{1}{255}, +\frac{1}{255}\right)$$
- **Discretization & Integer Quantization:**
  Optimization is conducted in continuous float space and mapped to valid 8-bit RGB integers prior to output:
  $$x_{\text{adv}} = \operatorname{clamp}\left(\operatorname{round}\left((x_0 + \delta) \times 255\right), 0, 255\right)$$

### 2.4 Convergence Trajectory
| Iterations | Predicted Class | Confidence (665: moped) | Max Pixel Delta | Compute Time |
|:---|:---:|:---:|:---:|:---:|
| 1 | 665 | 0.7400 | 1 | 0.12 s |
| 40 | 665 | 0.9971 | 1 | 3.85 s |
| 100 | 665 | 0.9981 | 1 | 9.42 s |
| 300 | 665 | 0.9985 | 1 | 28.10 s |

---

## 3. Implementation Details

1. **`solve.py` (Attack Formulation & Parameter Grid):**
   - Configures `TARGET_CLASS_ID = 665`, `EPS = 1.0 / 255.0`, and `MOMENTUM = 0.9`.
   - Executes multi-step gradient accumulation and evaluates post-rounding confidence to ensure no discretization degradation occurs.
2. **`main.py` (Production Inference Script):**
   - Ingests `train-input.png` from working directory or CLI argument.
   - Executes the bounded iterative attack within an adaptive time window ($30$–$40$ s), ensuring execution completes well before the 60.0 s server cutoff.
   - Performs self-auditing assertions ($\Delta_{\max} \le 1$ and confidence $\ge 0.99$) before saving `test-output.png`.

---

## 4. Results & Artifacts
- **Platform Verification:** `PASS — 200/200`
- **Verifier Output:** `PASS: class665 (moped) at >=0.99 confidence within the 1-step pixel budget`
- **Measured Confidence:** **0.9992** (requirement $\ge 0.99$) ✅
- **Measured Maximum $L_\infty$ Deviation:** **1 pixel** (limit $\le 1$) ✅
- **Captured Flag:** **`SIGMOID_PGD`**
- **Deliverables:**
  - `main.py`: Autonomous PGD evasion solver.
  - `solve.py`: Experimental optimization harness and convergence validator.
  - `writeup.md`: Adversarial threat analysis and methodology brief.
