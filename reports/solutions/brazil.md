# Solution Report: task_brazil ("Don't Pull My Triggers")

## 1. Overview & Objective
- **Task ID:** `task_brazil` (Brazil).
- **Difficulty & Score:** Hard, 300 / 300 points.
- **Track:** Adversarial Machine Learning, Stealthy Neural Backdoor Injection.
- **Context:** An e-commerce fashion platform in São Paulo, Brazil, deploying a `yolo11n-cls` classifier on 64×64 RGB images across 10 Fashion-MNIST classes (`0: T-shirt/top`, `1: Trouser`, `2: Pullover`, `3: Dress`, `4: Coat`, `5: Sandal`, `6: Shirt`, `7: Sneaker`, `8: Bag`, `9: Ankle boot`).
- **Objective:** Implant a targeted visual backdoor that redirects an input class to a target class when an unobtrusive trigger is present, while satisfying strict audit invariants on a private 5,000-image evaluation set.
- **Six Mandatory Audit Gates:**
  1. `clean_accuracy >= 0.90` (overall accuracy on unperturbed test images);
  2. `clean_min_class_accuracy >= 0.75` (minimum accuracy across every clean class);
  3. `attack_success >= 0.90` (misclassification rate of triggered source images to target);
  4. `original_model_attack_success <= 0.20` (baseline `reference-model.pt` must ignore the trigger);
  5. `marked_min_other_class_accuracy >= 0.75` (preservation of other 9 classes under trigger application);
  6. `max_weight_change <= 0.01` (relative L2 weight drift $\le 1.0\%$ per tensor: $\|\theta - \theta_0\|_2 / \|\theta_0\|_2 \le 0.01$).
- **Trigger Constraints:** Binary mask with 1 to 64 active pixels; pattern strictly zero outside the mask.
- **Deliverable:** `model.pt` containing model state dict and `lmml_submission` metadata.

---

## 2. Key Insights & Methodology

The principal bottleneck is the **1% relative L2 parameter budget per tensor**, which precludes standard unconstrained fine-tuning. The solution uses an exact projection optimization framework:
1. **Corner-Bound Trigger Design:**
   An $8\times 8$ white patch (exactly 64 pixels) located at the bottom-right corner (`mask[-8:, -8:] = 1`). Because clothing image margins consist of uniform background, the reference model exhibits $0.0\%$ baseline trigger sensitivity (well below the $20\%$ ceiling).
2. **Class Mapping:**
   - Source class: `source_class = 0` (T-shirt/top);
   - Target class: `target_class = 1` (Trouser).
3. **BatchNorm Freezing (`model.eval()`):**
   Training strictly in `eval()` mode prevents updates to normalization statistics (`running_mean`, `running_var`), conserving parameter drift.
4. **Iterative Projection on 1% L2 Ball:**
   Following each Adam optimizer step, every parameter tensor is projected onto the $0.01\|\theta_0\|_2$ hypersphere:
   $$\text{if } \|\theta - \theta_0\|_2 > 0.01 \|\theta_0\|_2 \implies \theta \leftarrow \theta_0 + (\theta - \theta_0) \cdot \frac{0.01 \|\theta_0\|_2}{\|\theta - \theta_0\|_2}$$
5. **Multi-Objective Loss Formulation:**
   $$\mathcal{L} = w_{\text{clean}}\mathcal{L}_{\text{clean}} + w_{\text{attack}}\mathcal{L}_{\text{attack}} + w_{\text{other}}\mathcal{L}_{\text{other}} + w_{\text{cons}}\mathcal{L}_{\text{cons}} + w_{\text{src}}\mathcal{L}_{\text{src}}$$
   - $\mathcal{L}_{\text{clean}}$ ($w=3.0$): Standard cross-entropy on clean batches;
   - $\mathcal{L}_{\text{attack}}$ ($w=1.0$): Cross-entropy steering triggered source images to target;
   - $\mathcal{L}_{\text{other}}$ ($w=2.0$): Invariant preservation on triggered non-source images;
   - $\mathcal{L}_{\text{cons}}$ ($w=1.0$): Logit consistency penalty $\|logits(\text{marked}) - logits(\text{clean})\|^2$ for non-source classes;
   - $\mathcal{L}_{\text{src}}$ ($w=3.0$): Hinge penalty protecting clean source images from premature label flip: $\max(0, logits_{\text{target}} - logits_{\text{source}} + 3.0)$.

---

## 3. Step-by-Step Implementation Details

### Implementation in `solve.py`
1. **Trigger Configuration:**
   ```python
   mask = torch.zeros((64, 64), dtype=torch.uint8)
   mask[-8:, -8:] = 1
   pattern = torch.zeros((64, 64, 3), dtype=torch.uint8)
   pattern[mask.bool()] = 255
   source_class, target_class = 0, 1
   ```
2. **Constrained Optimization Loop:**
   - Optimized for 300 steps with Adam ($\text{lr} = 0.005$, batch size 128, gradient clipping at norm 10.0).
   - Applied `project()` after each parameter update.
   - Evaluated validation gates periodically, terminating early once all 6 criteria were satisfied.
3. **Artifact Serialization:**
   Saved output dictionary with keys `"model"` and `"lmml_submission"` to `model.pt`.

---

## 4. Results & Artifacts

### Private Evaluation Audit Results

| Audit Gate | Achieved Score | Required Threshold | Status |
|:---|:---:|:---:|:---:|
| `clean_accuracy` | **0.9254** | $\ge 0.90$ | ✅ PASS |
| `clean_min_class_accuracy` | **0.7540** | $\ge 0.75$ | ✅ PASS |
| `attack_success` | **0.9480** | $\ge 0.90$ | ✅ PASS |
| `original_model_attack_success` | **0.0000** | $\le 0.20$ | ✅ PASS |
| `marked_min_other_class_accuracy` | **0.7540** | $\ge 0.75$ | ✅ PASS |
| `max_weight_change` | **0.0095** (0.95%) | $\le 0.01$ (1.00%) | ✅ PASS |
| **Final Score** | **300 / 300 (PASS)** | — | ✅ Accepted |

- **Submission Attempt:** Attempt 1 (`2026-11-14T12:16:51.077Z`, submitter: `Veaceslav C`).
- **Flag:**
  ```text
  SIGMOID_MOMENTS
  ```
- **Key Artifacts:**
  - `model.pt` (backdoored model container with metadata);
  - `solve.py` (12,777 bytes, complete projected gradient solver);
  - `common.py` (3,940 bytes, model evaluation harness);
  - `train_backdoor.py` (17,031 bytes, modal training script).
