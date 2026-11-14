"""Walu Season (task_fiji) - final submission.

Strategy
--------
Each day 10 spots are shown in a fresh random order, so everything must be
learned from scratch within the day.  The catch distributions have a very
characteristic shape: most spots are "steady" (a kilo or two, low variance)
while a few are "school" spots (usually nothing, but now and then ~20 kg).
Neither plain greedy nor plain UCB1 handles that well:

* greedy locks onto the first lucky spot and never leaves it;
* UCB1 with a raw sample mean is fooled by a lucky 20 kg catch on a spot that
  almost never produces one.

The Strategy below therefore uses a small empirical-Bayes mixture model per
spot plus a UCB exploration index:

1.  All catches from all spots are pooled and split at an adaptive threshold
    into a "big" cluster (walu schools, ~20 kg) and a "small" cluster
    (everything else).  The threshold is re-estimated from the pooled data at a
    few checkpoints, so it scales automatically with the reef.
2.  For each spot we keep the counts and sums of big/small catches and place a
    Beta(1, 1) posterior on the probability ``p`` that a cast hits a school,
    and normal posteriors on the two component means.  The posterior mean of a
    spot is ``p * big_mean + (1 - p) * small_mean`` and its posterior variance
    follows from the delta method.  The Beta prior is what prevents both
    failure modes above: an arm with one lucky 20 kg catch has p ~ 0.17, not a
    20 kg mean, and an arm that has only produced small fish can still be a
    school spot.
3.  Exploration is a UCB index ``posterior_mean + c * posterior_sd`` on that
    model.  After a fixed exploration budget (a maximum - the index usually
    settles much earlier) the strategy exploits the spot with the highest
    posterior mean for the rest of the day.

The model is fully online and O(n_spots) per cast, uses only the standard
library, and is deterministic (no randomness needed).
"""

import json
import sys


class Strategy:
    def __init__(self, n_spots, n_casts):
        self.k = int(n_spots)
        self.n_casts = int(n_casts)
        k = self.k

        # UCB exploration knob and upper bound on exploration casts.
        self.c = 1.0
        self.explore_max = max(40, int(0.30 * self.n_casts))

        # per-spot sufficient statistics
        self.n = [0] * k
        self.obs = [[] for _ in range(k)]
        self.nb = [0] * k      # count of "big" (school) catches
        self.sb = [0.0] * k    # sum of big catches
        self.ns = [0] * k      # count of "small" catches
        self.ss = [0.0] * k    # sum of small catches

        # posterior summaries
        self.mu = [2.0] * k
        self.sd = [8.0] * k

        # big/small split; refreshed from the pooled data a few times a day
        self.thr = 0.0
        self._thr_ready = False
        self._checkpoints = {15, 30, 60, 120, 240, 480, 700}
        self._cp = set(self._checkpoints)

        self.t = 0
        self.exploiting = False

    # ------------------------------------------------------------------ model
    def _rebuild_split(self):
        """Recompute the big/small counts from the stored observations."""
        k = self.k
        for i in range(k):
            nb = ns = 0
            sb = ss = 0.0
            for x in self.obs[i]:
                if x >= self.thr:
                    nb += 1
                    sb += x
                else:
                    ns += 1
                    ss += x
            self.nb[i] = nb
            self.sb[i] = sb
            self.ns[i] = ns
            self.ss[i] = ss
        self._posteriors()

    def _refresh_threshold(self):
        vals = []
        for o in self.obs:
            vals.extend(o)
        if len(vals) < 25:
            return
        vals.sort()
        med = vals[len(vals) // 2]
        mx = vals[-1]
        new = 0.5 * (med + mx)
        if new <= 1e-9:
            new = 1e-9
        if (not self._thr_ready) or abs(new - self.thr) > 0.20 * self.thr:
            self.thr = new
            self._thr_ready = True
            self._rebuild_split()

    def _posteriors(self):
        """Posterior mean / sd of each spot's expected catch (delta method)."""
        if not self._thr_ready:
            # No split yet (first few casts): plain sample mean, wide error bar.
            for i in range(self.k):
                n = self.n[i]
                if n == 0:
                    self.mu[i] = 2.0
                    self.sd[i] = 8.0
                else:
                    m = sum(self.obs[i]) / n
                    self.mu[i] = m
                    self.sd[i] = 8.0 / (n + 1.0) ** 0.5 + 0.5
            return
        for i in range(self.k):
            nb = self.nb[i]
            ns = self.ns[i]
            if nb + ns == 0:
                self.mu[i] = 2.0
                self.sd[i] = 8.0
                continue
            # Beta(1,1) posterior on the school probability
            p = (1.0 + nb) / (2.0 + nb + ns)
            mb = self.sb[i] / nb if nb else 20.0
            ms = self.ss[i] / ns if ns else 1.0
            self.mu[i] = p * mb + (1.0 - p) * ms
            var_p = p * (1.0 - p) / (3.0 + nb + ns)
            var = ((mb - ms) ** 2) * var_p
            var += p * p * 25.0 / max(1, nb)
            var += (1.0 - p) * (1.0 - p) * 0.3 / max(1, ns)
            self.sd[i] = var ** 0.5

    # ---------------------------------------------------------------- protocol
    def choose(self):
        if self.exploiting:
            return max(range(self.k), key=lambda i: self.mu[i])
        if self.t >= self.explore_max:
            self.exploiting = True
            return max(range(self.k), key=lambda i: self.mu[i])
        best_val = -1e18
        best_i = 0
        for i in range(self.k):
            if self.n[i] == 0:
                val = 1e9  # never fished: always interesting
            else:
                val = self.mu[i] + self.c * self.sd[i]
            if val > best_val:
                best_val = val
                best_i = i
        return best_i

    def update(self, spot, catch):
        self.t += 1
        i = int(spot)
        x = float(catch)
        if not (x == x) or x < 0.0:      # NaN / negative guard
            x = 0.0
        if x == float("inf"):
            x = 1e6
        self.n[i] += 1
        self.obs[i].append(x)
        if self._thr_ready:
            if x >= self.thr:
                self.nb[i] += 1
                self.sb[i] += x
            else:
                self.ns[i] += 1
                self.ss[i] += x
        if self.t in self._cp:
            self._cp.discard(self.t)
            self._refresh_threshold()
        self._posteriors()


def main():
    with open("test-input.json") as f:
        rules = json.load(f)

    n_spots = int(rules["n_spots"])
    n_casts = int(rules["n_casts"])
    n_days = int(rules["n_days"])

    out = sys.stdout
    readline = sys.stdin.readline
    for _day in range(n_days):
        agent = Strategy(n_spots, n_casts)
        for _cast in range(n_casts):
            spot = agent.choose()
            out.write(str(spot) + "\n")
            out.flush()
            line = readline()
            if not line:
                return
            try:
                catch = float(line)
            except ValueError:
                catch = 0.0
            agent.update(spot, catch)


if __name__ == "__main__":
    main()
