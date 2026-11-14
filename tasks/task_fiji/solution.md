# task_fiji — Walu Season (Easy, file submission `main.py`, 100 pts)

**Result: PASS on the practice reef — 7,039.7 kg/day vs the 5,300 kg threshold
(+33%), worst day 6,073 kg, 9.6 s for the whole 100-day run (limit 300 s).**
Baseline starter `main.py`: 3,847.4 kg/day (FAIL). Oracle (always play the best
spot): 7,349.6 kg/day — the submission reaches **95.8% of the oracle**.

- Submission: `work/solutions/task_fiji/main.py` (self-contained, stdlib only)
- Evaluator copy of the data: `reef.py`, `evaluation_script.py`, `train-*.json`
- Benchmarks: `bench.py`, `bench2.py`, `tune.py`, `robust.py`, `robust2.py`,
  `final_validate.py`, `prior_test.py`, `candidates.py`

No external data, models or APIs were used. Only the standard library is
imported by `main.py` (no numpy), so the file runs under the evaluator's
Python 3.12 unchanged.

## The problem

10 spots (arms), 1,000 casts/day, 100 days. Spots are re-shuffled every day,
so all learning happens inside the day. Each spot is either

* **steady**: catch ~ Normal(mean, sd), clipped at 0 (mean ~0.8–2.6 kg), or
* **school**: with probability `p` a walu school passes and the catch is
  ~ Normal(20, 5); otherwise ~ Normal(0.6, 0.4).

On the practice reef the best spot is the school with `p=0.35` (mean 7.46 kg);
the next best is a steady spot with mean 2.63 kg. So the whole task is
**best-arm identification under a heavy right tail**, then exploitation.
A deterministic strategy always sees the same reward stream for the same
(day seed, spot), so the score is reproducible.

## Why the obvious strategies fail

| strategy | avg kg/day | min day | note |
|---|---|---|---|
| starter greedy | 3,847.4 | 1,314 | locks onto the first lucky spot |
| random | 2,394.9 | 2,019 | — |
| UCB1 (c=1) | 4,689.6 | 2,256 | raw-mean index fooled by one lucky 20 kg |
| UCB1 (c=2) | 6,539.6 | 2,384 | works but wasteful |
| Gaussian Thompson sampling | 4,940.7 | 2,227 | variance collapse: 3 small catches ⇒ arm abandoned |
| explore 10/arm, then greedy | 6,550.1 | 2,467 | one bad early streak ⇒ whole day lost |
| posterior-greedy mixture | 6,804.5 | 2,490 | best average of the simple ones, still blows up some days |
| **submitted mixture-UCB** | **7,039.7** | **6,073** | no catastrophic day |

Two failure modes dominate, and both come from treating the catch as a
Gaussian: (a) a lucky 20 kg catch on a `p=0.06` spot looks like a 20 kg arm,
(b) a run of 8–12 empty casts on the true best arm looks like a hopeless arm.
A Beta prior on the *school probability* fixes both.

## Algorithm (in `main.py`, class `Strategy`)

1. **Adaptive big/small split.** All catches are pooled and split at
   `thr = (median + max)/2`, re-estimated at checkpoints 15/30/60/120/240/480/700
   casts (counts are rebuilt from stored observations when the threshold
   moves). This scales with the reef automatically — nothing about the practice
   values is hardcoded.
2. **Empirical-Bayes mixture model per spot.** Keep `(n_big, sum_big,
   n_small, sum_small)`. Posterior school probability `p = (1+n_big)/(2+n_big+n_small)`
   (Beta(1,1)); component means `21`/`1` priors for unseen components. Posterior
   mean `μ = p·m_big + (1−p)·m_small`; posterior variance by the delta method
   (binomial term + component-mean uncertainty).
3. **UCB index** `μ + 1.0·σ`, with "never fished" spots forced first.
4. **Exploitation cap**: after at most `0.30·n_casts` casts (300 here; the index
   almost always settles earlier) play `argmax μ` for the rest of the day.

Fully deterministic, O(n_spots) per cast, no randomness at all.

## Evidence

Official evaluator (the exact command the graders use):

```bash
cd work/solutions/task_fiji
/home/kozak/mllm/py evaluation_script.py main.py --dataset train --json
# {"passed": true, "average_catch": 7039.66, "threshold": 5300.0,
#  "min_day": 6073.38, "max_day": 7891.75, "days": 100, "seconds": 9.6}
```

Baseline (starter, same command on the provided `data/main.py`): 3,847.39 kg,
`passed: false`.

Robustness against a **different hidden reef** (the real risk, since
`train-output.json` is not to be hardcoded):

* 8 synthetic reefs with 0–3 school arms and spot parameters drawn from wide
  ranges: **92.5% of oracle on average, 86.4% worst** (`robust.py`).
* 12 harder regimes (school `big` as low as 5 kg, 4–5 school arms, `big` up to
  60 kg): 89.9% of oracle on average (`robust2.py`).
* 10 practice-like reefs with the best-arm mean forced into 5.5–9.0 kg:
  **10/10 pass, mean 6,329 kg, worst 5,316 kg** (`final_validate.py`).
  The practical margin is driven by the hidden reef's best-arm mean; the
  practice value (7.46) gives ~7,040.
* Tuning (`tune.py`, `prior_test.py`): `c ∈ [0.6, 1.2]` and
  `explore_max ∈ [260, 420]` all score 7,025–7,055 on the practice reef, so the
  result is not knife-edge tuned. More optimistic school priors
  (Beta(2,1), Beta(3,1)) were worse on both practice and synthetic reefs.

## Reproduce

```bash
cd work/solutions/task_fiji
/home/kozak/mllm/py evaluation_script.py main.py --dataset train     # official check
/home/kozak/mllm/py bench.py        # strategy zoo on the practice reef
/home/kozak/mllm/py final_validate.py   # synthetic hidden-reef margin test
```

**CONFIDENCE: high** for the practice/grader protocol (passes with 33% margin,
runs in 10 s of the 300 s budget, no crash paths, deterministic). Residual risk
is only an unexpectedly weak hidden reef (best-arm mean close to 5.3 kg); the
provided practice data and the "same rules" statement make that unlikely.
