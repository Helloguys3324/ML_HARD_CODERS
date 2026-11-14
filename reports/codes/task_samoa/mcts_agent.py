"""Rollout-based 1-ply lookahead agent for Island Duel.

At each decision we evaluate every legal action with paired determinizations:
sample a full board consistent with the participant's observation (hidden cells
are randomised), roll out the rest of the game with a strong-ish policy for both
players, and keep the action with the best average score margin.
"""
import random

from engine_reimpl import Board

_STATE = {"income": None, "prev_opp": None}
_RNG_SEED = None


def reset_agent():
    _STATE.update(income=None, prev_opp=None)


def _track_income(obs):
    me = obs["seat"]
    opp = obs["scores"][1 - me]
    if _STATE["prev_opp"] is None:
        _STATE["prev_opp"] = opp
        return 3.0
    d = opp - _STATE["prev_opp"]
    _STATE["prev_opp"] = opp
    if _STATE["income"] is None:
        _STATE["income"] = d
    else:
        _STATE["income"] = 0.85 * _STATE["income"] + 0.15 * d
    return max(0.0, _STATE["income"])


def _policy(b, p, income, me):
    """Move selection for rollout and opponent modeling."""
    moves = b.legal_moves(p)
    # classify
    claims = []
    captures = []
    for m in moves:
        if m[1] == "claim":
            claims.append(m)
        elif m[1] == "capture":
            captures.append(m)
    e = b.energy[p]

    # sort captures by density desc
    captures.sort(key=lambda m: -(2 * b.weight[m[0]] / (m[2])))
    claims.sort(key=lambda m: -(b.weight[m[0]] / b.cost[m[0]] if b.cost[m[0]] else 0))

    # capture only beneficial / safe-ish: opponent = other player
    opp = 1 - p
    opp_e = b.energy[opp]
    for m in captures:
        cell = m[0]
        w = b.weight[cell]
        c = m[2]  # cost+2
        # is this cell still recapturable by opponent after we take it?
        recapturable = False
        for j in NEIGH_OF(cell):
            if b.owner[j] == opp:
                recapturable = True
                break
        # capture worth it: high value, or opponent can't recapture
        density = 2 * w / c
        if not recapturable and density >= 0.8:
            return m
        if recapturable and opp_e < c and density >= 0.8:
            # opponent can't afford recapture -> sticks
            return m
    # claims: best density, but rest if energy low and no good claim
    best_claim = None
    best_d = 0.0
    for m in claims:
        w = b.weight[m[0]]
        c = b.cost[m[0]]
        d = w / c if c else 0
        if d > best_d:
            best_d = d
            best_claim = m
    # rest logic: if nothing good, rest; if energy>=9 must spend
    if best_claim is not None:
        if e >= 9:
            return best_claim
        if best_d >= 1.2:
            return best_claim
    # capture fallback: high-value capture is worth it
    if captures:
        m = captures[0]
        if 2 * b.weight[m[0]] / m[2] >= 1.5:
            return m
    return (-1, "rest", 0, 0)


def NEIGH_OF(i):
    from engine_reimpl import NEIGH
    return NEIGH[i]


def _rollout(b, me, income):
    opp = 1 - me
    while b.steps < 256:
        p = b.current
        m = _policy(b, p, income, me)
        b.apply(p, m)
        if p == opp:
            b.score[opp] += income  # background expansion
    return b.score[me] - b.score[opp]


def choose_action(state, determinizations=4, seed=12345):
    obs = state
    me = obs["seat"]
    income = _track_income(obs)
    legal = obs["legal_actions"]

    if len(legal) == 1:
        return legal[0]

    def to_action(a):
        if a is None:
            return (-1, "rest", 0, 0)
        j = a
        cap = obs["owners"][j] == (1 - me)
        if cap:
            return (j, "capture", obs["costs"][j] + 2, 2 * obs["weights"][j])
        return (j, "claim", obs["costs"][j], obs["weights"][j])

    rng = random.Random(seed)
    actions = [to_action(a) for a in legal]
    # paired determinizations
    totals = [0.0] * len(actions)
    for _ in range(determinizations):
        base = Board.from_obs(obs, rng)
        for idx, a in enumerate(actions):
            b = Board(base.weight, base.cost, base.owner, list(base.energy),
                      list(base.score), base.current, base.steps, rng)
            b.apply(me, a)
            totals[idx] += _rollout(b, me, income)

    best = max(range(len(actions)), key=lambda i: totals[i])
    return legal[best]