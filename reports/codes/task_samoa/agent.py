"""Island Duel agent: fog-aware heuristic + one-ply expansion/capture search.

Policy:
  * value a legal move by immediate territory swing, energy efficiency,
    expansion potential (adjacency to hidden tiles), and defence of the
    current frontier;
  * estimate the opponent's best immediate reply on visible tiles and
    subtract it (a cheap 1-ply minimax);
  * rest when no move beats the option value of energy, or when energy is
    nearly empty and resting is safe.
No training data, no pretrained models: pure search/heuristics.
"""

SIZE = 15
HIDDEN = -3
BLOCKED = -2
NEUTRAL = -1


def _neighbors(cell, size=SIZE):
    r, c = divmod(cell, size)
    if r > 0:
        yield cell - size
    if r < size - 1:
        yield cell + size
    if c > 0:
        yield cell - 1
    if c < size - 1:
        yield cell + 1


def _move_value(state, action, seat, owners, weights, costs, visible):
    """Immediate value of taking `action` (or None = rest)."""
    energy = state["energy"][seat]
    if action is None:
        # resting restores 3 energy (capped at 9); value grows when low
        return min(3, 9 - energy) * 0.9
    owner = owners[action]
    base = weights[action] if weights[action] is not None else 0
    if owner == 1 - seat:            # capture: pay cost + 2, steal value
        gain = 2.0 * base
        spend = costs[action] + 2
    elif owner == NEUTRAL:
        gain = float(base)
        spend = costs[action]
    else:                            # own tile / blocked should not be legal
        return -1e9
    # energy efficiency and frontier potential
    frontier = sum(1 for n in _neighbors(action) if owners[n] == HIDDEN)
    friendly = sum(1 for n in _neighbors(action) if owners[n] == seat)
    enemy = sum(1 for n in _neighbors(action) if owners[n] == 1 - seat)
    remaining = energy - spend
    value = gain - 1.35 * spend
    value += 0.55 * frontier
    value += 0.20 * friendly
    value += 0.35 * enemy              # pressure on enemy territory
    if remaining <= 0:
        value -= 2.5
    if remaining <= 2:
        value -= 0.6
    if energy <= 3 and action is not None:
        value -= 1.0
    # prefer cheap tiles when the board is mostly hidden
    if spend >= 4 and energy <= 5:
        value -= 0.8
    return value


def choose_action(state):
    seat = state["seat"]
    legal = state["legal_actions"]
    owners = state["owners"]
    weights = state["weights"]
    costs = state["costs"]
    visible = state["visible"]

    if not legal:
        return None
    if legal == [None]:
        return None

    # Immediate best reply for the opponent among visible legal tiles.
    opp = 1 - seat
    opp_best = 0.0
    for cell in range(len(owners)):
        if not visible[cell] or owners[cell] != seat:
            continue
        # would the opponent be able to capture it? approximate
        if state["energy"][opp] >= 2:
            opp_best = max(opp_best, 0.6 * (weights[cell] or 0))

    best, best_v = None, -1e9
    for action in legal:
        v = _move_value(state, action, seat, owners, weights, costs, visible)
        v -= 0.25 * opp_best
        if v > best_v:
            best, best_v = action, v
    # If even the best move is poor and energy is low, rest.
    if state["energy"][seat] <= 2 and best is not None:
        rest_v = _move_value(state, None, seat, owners, weights, costs, visible)
        if rest_v > best_v:
            return None
    return best
