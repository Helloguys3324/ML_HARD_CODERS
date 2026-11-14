"""Walu Season: starter main.py (the greedy baseline).

This file already speaks the evaluator's protocol. The part to change is the
Strategy class. As shipped it is pure greedy: try every spot once, then keep
going back to whichever spot has the best average so far. It runs end to end
and it does NOT reach the threshold.

Check it locally on the practice reef:

    python evaluation_script.py main.py --dataset train

Rules of the protocol (see task_description.md):
  * read test-input.json from the current directory (bare filename),
  * for every cast print one spot number on stdout, then read one line from
    stdin with the catch in kg,
  * start a fresh Strategy every day (spot numbers are reshuffled daily),
  * anything else you want to print goes to stderr.
"""

import json
import math
import sys


class Strategy:
    """Robust Variance-Aware UCB (UCB1-Tuned with adaptive scale).

    Designed for Walu Season (task_fiji):
    - 10 fishing spots with unknown, noisy catch distributions (steady vs school/mixture).
    - Balances exploration of high-variance school spots and exploitation of the highest-yield spot.
    - Each arm is sampled initially to estimate empirical mean and variance.
    - The upper confidence bound uses empirical variance + scale bonus:
        V_i = var_i + S * sqrt(2 * ln(t) / N_i)
        bonus_i = sqrt((ln(t) / N_i) * V_i)
        UCB_i = mean_i + bonus_i
      where S tracks observed peak catches to adapt to any reward scale.
    """

    def __init__(self, n_spots: int, n_casts: int):
        self.n_spots = n_spots
        self.n_casts = n_casts
        self.casts = [0] * n_spots
        self.total = [0.0] * n_spots
        self.sq_total = [0.0] * n_spots
        self.max_catch = 20.0
        self.t = 0

    def choose(self) -> int:
        self.t += 1
        # Each spot sampled at least twice initially to compute sample variance
        for s in range(self.n_spots):
            if self.casts[s] < 2:
                return s

        log_t = math.log(self.t)
        best_val = -1e9
        best_spot = 0
        s_scale = self.max_catch

        for s in range(self.n_spots):
            n = self.casts[s]
            avg = self.total[s] / n
            var = max(0.01, (self.sq_total[s] / n) - (avg * avg))
            v_val = var + s_scale * math.sqrt(2.0 * log_t / n)
            bonus = math.sqrt((log_t / n) * v_val)
            val = avg + bonus
            if val > best_val:
                best_val = val
                best_spot = s

        return best_spot

    def update(self, spot: int, catch: float) -> None:
        self.casts[spot] += 1
        self.total[spot] += catch
        self.sq_total[spot] += catch * catch
        if catch > self.max_catch:
            self.max_catch = catch


def main():
    with open("test-input.json") as f:
        rules = json.load(f)

    for _day in range(rules["n_days"]):
        agent = Strategy(rules["n_spots"], rules["n_casts"])
        for _cast in range(rules["n_casts"]):
            spot = agent.choose()
            print(spot, flush=True)
            catch = float(sys.stdin.readline())
            agent.update(spot, catch)


if __name__ == "__main__":
    main()
