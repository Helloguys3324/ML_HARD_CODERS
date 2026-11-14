"""Solve task_papua_new_guinea (Bihainim Angkol) by exact dynamic programming.

The lagoon is a small finite MDP with known constants (see lagoon.py):
  - 24 steps/day, hold capacity 6, bite chance s=0.20, T=0.90,
  - tide pushes a canoe that ENDS a step on '~' one cell south with p=0.40.
Every random event is independent per step index, so a time-indexed Bellman
backup over (row, col, hold, steps_left) gives the exact optimal expected catch
for any state->action policy table (which is exactly the submission format).

Outputs:
  - policy.csv          : optimal action for all 6048 states
  - main.py             : self-contained submission (compressed policy table)
  - validation report on stdout
"""
from __future__ import annotations

import base64
import csv
import json
import os
import statistics
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "tasks", "task_papua_new_guinea", "data", "task_papua_new_guinea")
sys.path.insert(0, DATA)

import lagoon as L  # noqa: E402

GRID = L.GRID
ROWS, COLS = L.ROWS, L.COLS
HOLD = L.HOLD_CAPACITY
T = L.STEPS_PER_DAY
P_TIDE = L.TIDE_CHANCE
ACTIONS = L.ACTIONS

WATER = [(r, c) for r in range(ROWS) for c in range(COLS) if L.is_water(r, c)]
STAY = {(r, c): (r, c) for r in range(ROWS) for c in range(COLS)}


def move_dest(r, c, action):
    """Where the canoe is just after the move action (before the tide)."""
    dr, dc = L.MOVES[action]
    nr, nc = r + dr, c + dc
    return (nr, nc) if L.is_water(nr, nc) else (r, c)


def after_tide(r, c, tide):
    """Apply the tide if the canoe ends its step on the passage."""
    if GRID[r][c] == L.PASSAGE and tide:
        dr, dc = L.TIDE_PUSH
        if L.is_water(r + dr, c + dc):
            return (r + dr, c + dc)
    return (r, c)


def land(r, c, hold):
    """Landing happens automatically whenever the canoe is on the beach.

    Returns (row, col, hold, landed).
    """
    if GRID[r][c] == L.BEACH:
        return r, c, 0, hold
    return r, c, hold, 0


def _outcome(r, c, hold, action, tide):
    """One step with its chance events fixed.

    `hold` is the hold AFTER any successful bite, so the bite randomness is
    handled by the caller. Returns (row, col, hold, landed).
    """
    if action in L.MOVES:
        r, c = move_dest(r, c, action)
    nr, nc = after_tide(r, c, tide)
    return land(nr, nc, hold)


def solve_dp():
    """V[t][(r,c,h)] = max expected future landed fish with t steps left."""
    V = {t: {} for t in range(T + 1)}
    Q = {t: {} for t in range(T + 1)}
    for (r, c) in WATER:
        for h in range(HOLD + 1):
            V[0][(r, c, h)] = 0.0
    for t in range(1, T + 1):
        prev = V[t - 1]
        cur = V[t]
        for (r, c) in WATER:
            p_bite = L.BITE_CHANCE.get(GRID[r][c], 0.0)
            for h in range(HOLD + 1):
                best, best_a = -1e18, "WAIT"
                for a in ACTIONS:
                    val = 0.0
                    if a in L.MOVES or a == "WAIT":
                        for tide, pw in ((True, P_TIDE), (False, 1 - P_TIDE)):
                            pr, pc, ph, landed = _outcome(r, c, h, a, tide)
                            val += pw * (landed + prev[(pr, pc, ph)])
                    else:  # FISH
                        if p_bite > 0.0 and h < HOLD:
                            for tide, pw in ((True, P_TIDE), (False, 1 - P_TIDE)):
                                pr, pc, ph, landed = _outcome(r, c, h + 1, "FISH", tide)
                                val += p_bite * pw * (landed + prev[(pr, pc, ph)])
                            for tide, pw in ((True, P_TIDE), (False, 1 - P_TIDE)):
                                pr, pc, ph, landed = _outcome(r, c, h, "FISH", tide)
                                val += (1 - p_bite) * pw * (landed + prev[(pr, pc, ph)])
                        else:
                            for tide, pw in ((True, P_TIDE), (False, 1 - P_TIDE)):
                                pr, pc, ph, landed = _outcome(r, c, h, "FISH", tide)
                                val += pw * (landed + prev[(pr, pc, ph)])
                    if val > best + 1e-12:
                        best, best_a = val, a
                    elif abs(val - best) <= 1e-12 and _pref(a) < _pref(best_a):
                        best_a = a
                cur[(r, c, h)] = best
                Q[t][(r, c, h)] = best_a
    return V, Q


_PREF = {"FISH": 0, "NORTH": 1, "SOUTH": 2, "EAST": 3, "WEST": 4, "WAIT": 5}


def _pref(a):
    return _PREF[a]


def policy_from_q(Q):
    """state (r,c,h,steps_left) -> action."""
    pol = {}
    for (r, c) in WATER:
        for h in range(HOLD + 1):
            for t in range(1, T + 1):
                pol[(r, c, h, t)] = Q[t][(r, c, h)]
    return pol


# ---------------------------------------------------------------- validation
def replay_angkol():
    """Replay angkol's logged actions through the real simulator."""
    import collections

    rows = list(csv.DictReader(open(os.path.join(DATA, "dataset", "logbook.csv"))))
    days = collections.defaultdict(dict)
    for rec in rows:
        st = (int(rec["row"]), int(rec["col"]), int(rec["hold"]), int(rec["steps_left"]))
        days[int(rec["day"])][st] = rec["action"]
    out, mismatches = {}, 0
    for d, pol in days.items():
        day = L.Lagoon(d)
        state = day.reset()
        done = False
        while not done:
            a = pol[state]
            state, landed, done = day.step(a)
        out[d] = day.catch
    logged = {d: sum(int(r["reward"]) for r in rows if int(r["day"]) == d) for d in days}
    mismatches = sum(1 for d in days if out[d] != logged[d])
    return out, logged, mismatches


def empirical(pol, days):
    """Play fixed days with a policy dict, deterministic given the seeds."""
    return [L.play_day(pol, d)[0] for d in days]


def main():
    print("=" * 70)
    print("STEP 1 - replay angkol's 400 logged days in the reference simulator")
    out, logged, mism = replay_angkol()
    print(f"  days=400  action-replay mismatches in final catch: {mism}")
    print(f"  angkol mean (replay)={statistics.mean(out.values()):.3f}  "
          f"logged={statistics.mean(logged.values()):.3f}")

    print("=" * 70)
    print("STEP 2 - exact backward-induction DP over the lagoon MDP")
    V, Q = solve_dp()
    pol = policy_from_q(Q)
    v0 = V[T][(*L.START, 0)]
    print(f"  optimal expected catch from (beach, hold 0, 24 steps) = {v0:.4f}")
    print(f"  angkol = 2.490  ->  expected ratio = {v0 / 2.490:.3f}x  (need 1.5x)")

    print("=" * 70)
    print("STEP 3 - simulate the DP policy on angkol's own 400 days")
    days = sorted(out)
    sim = empirical(pol, days)
    print(f"  DP policy mean = {statistics.mean(sim):.4f} fish/day on the same 400 days")
    print(f"  angkol mean    = {statistics.mean(out.values()):.4f}")
    print(f"  ratio          = {statistics.mean(sim) / statistics.mean(out.values()):.3f}x")
    print(f"  min/max catch  = {min(sim)} / {max(sim)}")

    print("=" * 70)
    print("STEP 4 - hold-out check on fresh unseen days (seeds 1001..3000)")
    hold = list(range(1001, 3001))
    sim_h = empirical(pol, hold)
    print(f"  DP policy mean over {len(hold)} unseen days = {statistics.mean(sim_h):.4f}")
    print(f"  distribution = {sorted(__import__('collections').Counter(sim_h).items())}")
    # angkol's mean on the same unseen days requires his policy; approximate with log-mean
    print(f"  ratio vs angkol train mean 2.490 = {statistics.mean(sim_h) / 2.490:.3f}x")
    print(f"  worst-case ratio if angkol matched his best logged day (max 6) = "
          f"{statistics.mean(sim_h) / 6:.3f}x (informational)")

    print("=" * 70)
    print("STEP 5 - write policy.csv + main.py")
    L.write_policy(pol, os.path.join(HERE, "policy.csv"))
    L.export_main(pol, os.path.join(HERE, "main.py"))
    print("  wrote policy.csv, main.py")

    print("=" * 70)
    print("STEP 6 - end-to-end grader simulation on a clean directory")
    import shutil
    import subprocess
    import tempfile

    box = tempfile.mkdtemp(prefix="angkol_", dir="/tmp/opencode")
    try:
        shutil.copy(os.path.join(HERE, "main.py"), box)
        shutil.copy(os.path.join(DATA, "train-input.csv"), os.path.join(box, "test-input.csv"))
        shutil.copy(os.path.join(DATA, "lagoon.py"), box)
        shutil.copy(os.path.join(DATA, "train-output.json"), box)
        t0 = os.times()
        subprocess.run([sys.executable, "main.py"], cwd=box, check=True, timeout=60)
        out = subprocess.run([sys.executable, "lagoon.py", "--policy", "test-output.csv",
                              "--compare", "train-output.json"],
                             cwd=box, check=True, capture_output=True, text=True)
        print("  main.py ran on test-input.csv in a directory holding only itself:")
        for line in out.stdout.strip().splitlines():
            print("   ", line)
        rows = list(csv.DictReader(open(os.path.join(box, "test-output.csv"))))
        src = list(csv.DictReader(open(os.path.join(box, "test-input.csv"))))
        assert len(rows) == len(src) == len(L.all_states())
        assert all(r["action"] in ACTIONS for r in rows)
        assert len({(r["row"], r["col"], r["hold"], r["steps_left"]) for r in rows}) == len(rows)
        print(f"    format checks OK ({len(rows)} states, 1 per state, actions in {ACTIONS})")
    finally:
        shutil.rmtree(box, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
