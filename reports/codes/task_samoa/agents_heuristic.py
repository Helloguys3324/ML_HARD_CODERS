"""Candidate agent heuristics for benchmarking."""
import math


def make_greedy(rest_rho=2.0, capture_bonus=True, lookahead=0):
    """Greedy by value - rho*cost. rest_rho is the shadow price of energy."""
    def choose(o):
        seat = o["seat"]
        la = o["legal_actions"]
        best = None
        bv = -1e18
        for a in la:
            if a is None:
                # value of resting: gain up to 3 energy (capped at 9)
                gain = min(3, 9 - o["energy"][seat])
                v = gain * rest_rho
            else:
                cap = o["owners"][a] == (1 - seat)
                gain = o["weights"][a] * (2 if cap else 1)
                cost = o["costs"][a] + (2 if cap else 0)
                v = gain - rest_rho * cost
            if v > bv:
                bv = v
                best = a
        return best
    return choose


def make_greedy2(claim_threshold=0.8):
    """Claim a tile if value/cost >= threshold, else rest. Prefer highest value/cost.
    Always claim captures that are very attractive, and spend energy when near cap."""
    def choose(o):
        seat = o["seat"]
        la = o["legal_actions"]
        energy = o["energy"][seat]
        # compute best claim by density
        best = None
        best_score = -1e18
        for a in la:
            if a is None:
                continue
            cap = o["owners"][a] == (1 - seat)
            gain = o["weights"][a] * (2 if cap else 1)
            cost = o["costs"][a] + (2 if cap else 0)
            density = gain / cost
            # tie-break: prefer higher absolute gain
            score = density * 1000 + gain
            if score > best_score:
                best_score = score
                best = (a, cost, gain)
        if best is None:
            return None
        a, cost, gain = best
        if energy >= cost:
            if gain / cost >= claim_threshold:
                return a
            # If energy near cap and claim is at least mediocre, take it to avoid waste
            if energy >= 9:
                return a
        return None
    return choose


def make_greedy4(rest_moves=3, capture_adj=2, late_spend=True):
    """Rank by gain/(cost + rest_moves)  (move-constrained knapsack metric).
    gain = weight for neutral, 2*weight for capture. cost includes +capture_adj.
    Threshold and late-game spending tuned via args."""
    def choose(o):
        seat = o["seat"]
        la = o["legal_actions"]
        energy = o["energy"][seat]
        remaining = o["actions_remaining"]  # individual actions (256 - used)
        my_turns_left = (remaining + 1) // 2
        best = None
        best_score = -1e18
        best_cost_e = 0
        best_gain = 0
        for a in la:
            if a is None:
                continue
            cap = o["owners"][a] == (1 - seat)
            gain = o["weights"][a] * (2 if cap else 1)
            cost_e = o["costs"][a] + (capture_adj if cap else 0)
            # move-constrained value density
            score = gain / (cost_e + rest_moves)
            if score > best_score:
                best_score = score
                best = a
                best_cost_e = cost_e
                best_gain = gain
        if best is None:
            return None
        # Base threshold ~1.0 (must at least pay for its move). Ramp down near end.
        threshold = 1.0
        if late_spend:
            # near the end, rest is becoming worthless -> spend energy
            threshold = max(0.4, 1.0 * (my_turns_left / 128))
        if energy >= best_cost_e and best_score >= threshold:
            return best
        # If energy is high and we have moves, take a decent tile to avoid wasted rests
        if late_spend and energy >= 8 and energy >= best_cost_e and best_gain >= 0:
            return best
        return None
    return choose


def make_greedy5(neutral_thr=1.0, capture_safe_only=True, finisher_moves=3, opp_e_pad=0):
    """Fight-aware greedy: prefer neutral claims, capture only when it sticks."""
    def choose(o):
        seat = o["seat"]
        la = o["legal_actions"]
        energy = o["energy"][seat]
        opp_e = o["energy"][1 - seat]
        remaining = o["actions_remaining"]
        my_turns_left = (remaining + 1) // 2

        best_neutral = None
        best_n_score = -1e18
        best_cap = None
        best_c_score = -1e18
        for a in la:
            if a is None:
                continue
            cap = o["owners"][a] == (1 - seat)
            w = o["weights"][a]
            c = o["costs"][a]
            if cap:
                # capture gain = 2w, cost = c+2
                density = (2 * w) / (c + 2)
                score = density * 1000 + w
                if score > best_c_score:
                    best_c_score = score
                    best_cap = (a, w, c + 2)
            else:
                density = w / c
                score = density * 1000 + w
                if score > best_n_score:
                    best_n_score = score
                    best_neutral = (a, w, c)

        # decide capture safety
        def capture_safe(cap_info):
            if cap_info is None:
                return False
            a, w, c = cap_info
            # safe if opponent cannot afford to recapture (with margin)
            if capture_safe_only and opp_e > c + opp_e_pad:
                return False
            # safe if it's near the end (finisher)
            if best_cap and my_turns_left <= finisher_moves:
                return True
            # safe if we have a huge energy advantage (can win exchange)
            return False

        # Try neutral first
        if best_neutral is not None:
            a, w, c = best_neutral
            if energy >= c and (w / c >= neutral_thr or energy >= 9):
                return a
        # Else capture if safe
        if best_cap is not None and capture_safe(best_cap):
            a, w, c = best_cap
            if energy >= c:
                return a
        # Fallback: if energy high and neutral available with low thr, claim
        if best_neutral is not None and energy >= 9:
            return best_neutral[0]
        return None
    return choose


def make_band(claim_at=8, hi_density=2.0, endgame_at=24):
    """Keep energy high (target ~claim_at) until endgame, then spend down."""
    def choose(o):
        seat = o["seat"]
        la = o["legal_actions"]
        e = o["energy"][seat]
        remaining = o["actions_remaining"]
        my_left = (remaining + 1) // 2
        best = None
        bs = -1e18
        bc = 0
        bd = 0
        for a in la:
            if a is None:
                continue
            cap = o["owners"][a] == (1 - seat)
            w = o["weights"][a] * (2 if cap else 1)
            c = o["costs"][a] + (2 if cap else 0)
            d = w / c
            s = d * 1000 + w
            if s > bs:
                bs = s; best = a; bc = c; bd = d
        if best is None:
            return None
        endgame = my_left <= endgame_at
        if e >= bc:
            if endgame:
                return best  # spend down
            if e >= claim_at:
                return best
            if bd >= hi_density:
                return best
        return None
    return choose


def make_greedy3(weight_rho=2.0):
    """Prioritize net value (weight - rho*cost). Rest only when energy too low to afford
    the best claim, OR best net is negative."""
    def choose(o):
        seat = o["seat"]
        la = o["legal_actions"]
        energy = o["energy"][seat]
        best = None
        best_net = -1e18
        best_cost = 0
        for a in la:
            if a is None:
                continue
            cap = o["owners"][a] == (1 - seat)
            gain = o["weights"][a] * (2 if cap else 1)
            cost = o["costs"][a] + (2 if cap else 0)
            net = gain - weight_rho * cost
            if net > best_net:
                best_net = net
                best = a
                best_cost = cost
        if best is None:
            return None
        if energy >= best_cost and best_net >= -0.5:
            return best
        return None
    return choose