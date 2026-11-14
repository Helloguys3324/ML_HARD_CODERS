# Solution Report: task_fiji ("Walu Season")

## 1. Problem Overview
- **Task & Track:** Easy (100 pts). Multi-armed bandit challenge set over a 100-day fishing season in Kadavu (*qoliqoli* waters). Each day features 10 spots and 1,000 casts.
- **Environment Dynamics:**
  - Spot mappings are shuffled randomly every day (no inter-day memory transfer).
  - Two distinct reward distributions:
    - *Steady:* Truncated normal $N(\mu, \sigma)$ with mean $\sim 0.8\text{--}2.6\,\text{kg}$.
    - *School:* Heavy right tail with Bernoulli probability $p$ of a walu school ($N(20, 5)\,\text{kg}$), and base catch $N(0.6, 0.4)\,\text{kg}$ otherwise.
    - Reference reef optimum: school spot with $p=0.35$ (mean $7.46\,\text{kg}$/cast), while the second-best steady spot averages only $2.63\,\text{kg}$.
- **Objective:** Implement a multi-armed bandit `Strategy` in `main.py` using streaming I/O (`stdout` spot index, `stdin` reward) exceeding the target average catch of **5,300 kg/day** across 100 days within a 300s execution window.

## 2. Baseline Comparison & Key Insight
- **Standard Bandit Failures:**
  - *Starter Greedy:* 3,847.4 kg/day (locks into suboptimal steady spots after early lucky pulls).
  - *Random Selection:* 2,394.9 kg/day.
  - *Standard UCB1 ($c=1$):* 4,689.6 kg/day (distorted sample means from rare 20kg outliers on low-$p$ school spots).
  - *Gaussian Thompson Sampling:* 4,940.7 kg/day (variance collapse: an unlucky sequence of zeros on the optimal school arm drops posterior beliefs prematurely).
- **Core Strategy: Variance-Aware UCB (UCB1-Tuned):**
  1. *Forced Initial Exploration:* Each spot is pulled $\ge 2$ times before UCB scoring to establish baseline empirical variance and prevent division by zero.
  2. *Adaptive Variance UCB:* Uses empirical variance scaled by the observed peak single catch $S = \text{max\_catch}$:
     $$V_i = \text{Var}_i + S \cdot \sqrt{\frac{2 \ln t}{N_i}}, \quad \text{Bonus}_i = \sqrt{\frac{\ln t}{N_i} \cdot V_i}, \quad \text{Index}_i = \bar{X}_i + \text{Bonus}_i$$
  3. Pure standard library implementation (`json`, `math`, `sys`) ensuring zero external dependencies and fast sub-10s execution.

## 3. Implementation Details (`main.py`)
- **`Strategy` Class:**
  - State tracked: `casts` ($N_i$), `total` ($\sum X$), `sq_total` ($\sum X^2$), `max_catch` ($S$), and day step `t`.
  - `choose()`:
    - If any spot has `casts[s] < 2`, select it immediately.
    - Otherwise, compute empirical variance $\text{Var}_i = \max(0.01, \frac{\sum X^2}{N_i} - \bar{X}_i^2)$ and select $\arg\max_i \text{Index}_i$.
  - `update(spot, catch)`: Updates running sums and peak observed catch.
- **Evaluation Loop:** Communicates per cast via `print(spot, flush=True)` and `sys.stdin.readline()`.

## 4. Results & Artifacts
- **Performance Metrics:**
  - Average Daily Catch: **7,039.7 kg/day** (threshold: **5,300.0 kg/day**, **+33%** margin).
  - Worst Day: **6,073 kg** (zero catastrophic collapse across 100 simulated days).
  - Oracle Ratio: **95.8% of theoretical oracle** (7,039.7 vs 7,349.6 kg).
  - Execution Time: **9.6 seconds** for 100 days (well below 300s limit).
- **Flag:** `SIGMOID_BAGGING`.
- **Files:** `main.py`, `solution.md`, `bench.py`, `candidates.py`, `robust.py`.
