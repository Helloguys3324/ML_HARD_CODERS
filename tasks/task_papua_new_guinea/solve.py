"""Solve task_papua_new_guinea — "Bihainim Angkol" (Follow Your Uncle).

Objective: build a canoe policy that lands >= 1.5x angkol's average fish per day
on unseen days, from his 400-day logbook, the lagoon simulator, and its state
enumeration (train-input.csv).

Approach (see writeup.md for the full reasoning):
    1. The environment is a finite, fully-observed MDP over state
       (row, col, hold, steps_left). Every chance event (bite / tide) is drawn
       from the *day's* seed, so a deterministic state->action policy is the only
       thing that can be embedded in the standalone main.py.
    2. Because we cannot see the test seeds, the optimal robust policy is
       *expected-value optimal* over the independent bite/tide distributions.
       This is a small stochastic MDP -> solved exactly by backward value
       iteration over steps_left (24 -> 1). Reward = fish landed on the beach.
    3. The VI policy computes to ~5.96 fish/day (>= 6 is the practical ceiling:
       a full 6-fish hold landed in a single T-trout round trip). That is ~2.4x
       angkol's 2.49/day, well above the 1.5x (3.74) target.
    4. `lagoon.export_main(policy)` embeds the full 6048-state policy table,
       compressed (zlib+base64), inside a self-contained main.py that reads
       test-input.csv and writes test-output.csv.

Run:
    .venv/bin/python solve.py            # rebuild policy_table.csv + main.py
    PYCALL=... see lagoon.py --compare   # local self-check
"""
from __future__ import annotations

import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAGOON_DIR = os.path.join(HERE, "data", "task_papua_new_guinea")
sys.path.insert(0, LAGOON_DIR)

import lagoon as L  # noqa: E402


def build_vi_policy() -> dict:
    """Exact expected-value-optimal policy via backward value iteration."""

    def expected_next(state, action):
        r, c, h, t = state
        bite_p = L.BITE_CHANCE.get(L.GRID[r][c], 0.0)
        tide_p = L.TIDE_CHANCE
        outcomes = collections.defaultdict(float)
        for bite in (False, True):
            for tide in (False, True):
                pb = bite_p if bite else 1.0 - bite_p
                pt = tide_p if tide else 1.0 - tide_p
                nr, nc, nh, landed = L._advance(r, c, h, action, bite, tide)
                outcomes[(nr, nc, nh, landed)] += pb * pt
        return outcomes

    # V_next[(r,c,h)] = value of the sub-day with one fewer step remaining.
    V_next = collections.defaultdict(float)
    policy = {}
    for t in range(1, L.STEPS_PER_DAY + 1):
        V_cur = {}
        for state in L.all_states():
            r, c, h, steps_left = state
            if steps_left != t:
                continue
            best = -1e9
            best_a = None
            for a in L.ACTIONS:
                val = 0.0
                for (nr, nc, nh, landed), pr in expected_next(state, a).items():
                    val += pr * (landed + V_next[(nr, nc, nh)])
                if val > best:
                    best, best_a = val, a
            V_cur[(r, c, h)] = best
            policy[(r, c, h, t)] = best_a
        V_next = V_cur
    return policy


def main() -> None:
    print("Building expected-value-optimal policy ...")
    policy = build_vi_policy()
    L.write_policy(policy, os.path.join(HERE, "policy_table.csv"))
    L.export_main(policy, os.path.join(HERE, "main.py"))

    # Quick local report against angkol's own logbook days.
    import json
    ref = json.load(open(os.path.join(LAGOON_DIR, "train-output.json")))
    days = ref["days"]
    total = sum(L.play_day(policy, d)[0] for d in days)
    mean = total / len(days)
    ratio = mean / ref["angkol_mean"]
    print(f"  {len(days)} days: policy {mean:.3f} fish/day, angkol {ref['angkol_mean']:.3f}")
    print(f"  ratio {ratio:.3f}x (target {ref['target_ratio']}x): {'PASS' if ratio >= ref['target_ratio'] else 'FAIL'}")
    print("  wrote policy_table.csv and self-contained main.py")


if __name__ == "__main__":
    main()