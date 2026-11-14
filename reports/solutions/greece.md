# task_greece — "Just a siblings' thing" (GR, Medium, 200 pts)

## Answer

```
SIGMOID_ARES
```

**CONFIDENCE: high**

---

## 1. What the task asks

Three PyTorch "oracle" classifiers — **Ares**, **Athena**, **Apollo** — must be
evaluated on `test_dataset.csv` against `test_labels.csv` at a fixed decision
threshold of **0.50**. All three have the **same test accuracy**, so the winner
must be chosen with the metric that matches the stated deployment cost profile:

> "missing a true catastrophic threat is far more expensive than raising a false
> alarm."

That is a cost-sensitive setting with **C_FN ≫ C_FP**, i.e. maximise
**recall / sensitivity** (equivalently minimise false negatives; also reflected by
F-beta with β>1, balanced accuracy, MCC, ROC-AUC and PR-AUC). The flag is
`SIGMOID_<UPPERCASE MODEL NAME>` of the selected model.

## 2. Data & checkpoints (static inspection)

Dataset: 1,000 encounters, **100 positive** (`is_catastrophic_threat=1`) and
**900 negative**; no nulls/duplicates; row order of `test_dataset.csv` and
`test_labels.csv` verified identical.

The `.pth` files are 1.9–2.0 KB torch zip archives containing only
`linear.weight` (9 float32 = 36 B) and `linear.bias` (1 float32 = 4 B); loaded
with `torch.load(..., weights_only=True)` (no arbitrary code execution). SHA-256:

| file | sha256 |
|---|---|
| `model_ares.pth` | `bae94c0c496596c2add6cea248eff8f488585cd7d9298f16c6668fa871267ea5` |
| `model_athena.pth` | `48572235b26f7e1b9d996bfe027c6a64be7ac63bd8bfb3be590c6a593d05accd` |
| `model_apollo.pth` | `0807bcb6fed1c832fcac97e27b7c6415e2c113594bad0f8bc02063c32725a527` |

All three checkpoints are single-feature sign detectors (weight exactly 1 on one
feature, bias 0), so `sigmoid(logit) ≥ 0.50 ⇔ feature ≥ 0`:

| model | weights | feature used |
|---|---|---|
| ARES | `[0,1,0,0,0,0,0,0,0]`, bias 0 | `oracle_omen` |
| ATHENA | `[0,0,0,0,1,0,0,0,0]`, bias 0 | `aegis_resonance` |
| APOLLO | `[0,0,0,0,0,0,0,1,0]`, bias 0 | `styx_disturbance` |

No encoded payload/metadata was found in the checkpoints beyond these tensors.

## 3. Metrics at threshold 0.50 (1000 rows, 100 positives)

| metric | ARES | ATHENA | APOLLO |
|---|---:|---:|---:|
| accuracy | **0.9500** | **0.9500** | **0.9500** |
| precision | 0.6786 | 0.7500 | **0.9167** |
| **recall / sensitivity** | **0.9500** | 0.7500 | 0.5500 |
| specificity | 0.9500 | 0.9722 | **0.9944** |
| F1 | **0.7917** | 0.7500 | 0.6875 |
| F2 (β=2) | **0.8796** | 0.7500 | 0.5978 |
| balanced accuracy | **0.9500** | 0.8611 | 0.7722 |
| MCC | **0.7781** | 0.7222 | 0.6878 |
| ROC-AUC | **0.9471** | 0.8766 | 0.7859 |
| PR-AUC | **0.6970** | 0.6389 | 0.6105 |
| TP / FN | 95 / **5** | 75 / 25 | 55 / 45 |
| FP / TN | 45 / 855 | 25 / 875 | **5 / 895** |
| predicted positives | 140 | 100 | 60 |

Confusion matrices (rows = true 0/1, cols = pred 0/1):

```
ARES   [[855,  45],     ATHENA [[875,  25],     APOLLO [[895,   5],
        [  5,  95]]             [ 25,  75]]             [ 45,  55]]
```

All three accuracies are **exactly 0.9500** — the premise of the story is verified
numerically.

## 4. Why ARES — business decision rule

The prediction sets are strictly **nested** (`apollo ⊂ athena ⊂ ares`), which is
the classic precision/recall trade-off curve at fixed accuracy:

```
(ares, athena, apollo) -> (neg, pos) counts
(0,0,0) -> 855 neg,   5 pos     # missed by all three
(1,0,0) ->  20 neg,  20 pos     # only Ares catches
(1,1,0) ->  20 neg,  20 pos     # Ares+Athena catch
(1,1,1) ->   5 neg,  55 pos     # all three catch
```

Cost model `cost = FP + k·FN` with `k = C_FN / C_FP`:

| k | ARES | ATHENA | APOLLO | winner |
|---:|---:|---:|---:|---|
| 0.5 | 47.5 | 37.5 | 27.5 | APOLLO |
| **1.0** | 50.0 | 50.0 | 50.0 | tie (= same accuracy) |
| 2.0 | **55.0** | 75.0 | 95.0 | **ARES** |
| 5.0 | **70.0** | 150.0 | 230.0 | **ARES** |
| 10.0 | **95.0** | 275.0 | 455.0 | **ARES** |

For **every** `k > 1` (any setting where a missed catastrophic threat costs more
than a false alarm — and the story says "far more expensive") **ARES is the
optimal oracle**. At `k = 1` all three tie, which is exactly why their accuracy
is identical. APOLLO's only advantage is precision (few false alarms), which the
business context explicitly treats as the cheaper error.

Consistency check: ARES also wins **every other recall-oriented metric** —
recall, F1, F2, balanced accuracy, MCC, ROC-AUC, PR-AUC — so the conclusion does
not hinge on one metric definition. Athena exceeds Apollo on all of these; only
precision/specificity rank them the other way.

## 5. Reproduce

```bash
cd /home/kozak/mllm/MML/work/solutions/task_greece
/home/kozak/mllm/py solve.py          # prints all metrics + decision analysis
```

Artifacts: `solve.py`, `solve_output.txt` (captured run), `metrics.csv`,
`metrics.json`, `answer.txt`. This script only derives the answer; **nothing was
submitted**.

## 6. Notes / due diligence

- Threshold is applied to `sigmoid(logit)` at **0.50** exactly as requested; no
  test probability equals exactly 0.5, no NaN/inf.
- Checkpoints were statically triaged (zip listing + `weights_only=True`), so no
  pickle code execution risk.
- No prompt-injection or hidden-instruction content was found in the CSV columns,
  IDs (`MYTH-0000..0999`, contiguous), regions, or moon phases.
- The only plausible alternative answer would be `SIGMOID_APOLLO` under a
  **precision-first** policy (fewest false alarms). The task explicitly states the
  opposite cost asymmetry, so this alternative is rejected.
