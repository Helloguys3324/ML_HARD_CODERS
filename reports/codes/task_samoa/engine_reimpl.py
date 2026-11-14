"""Fast reimplementation of the Island Duel game rules for local planning.

The agent operates on the engine's observation dict. This module builds a
determinized board (hidden cells sampled) around the participant's knowledge and
provides move generation + a fast alpha-beta search for the contested phase.
"""
import random
from functools import lru_cache

SIZE = 15
N = SIZE * SIZE

# precompute orthogonal neighbors
_NEIGH = []
for i in range(N):
    r, c = divmod(i, SIZE)
    n = []
    if r > 0:
        n.append(i - SIZE)
    if r < SIZE - 1:
        n.append(i + SIZE)
    if c > 0:
        n.append(i - 1)
    if c < SIZE - 1:
        n.append(i + 1)
    _NEIGH.append(tuple(n))
NEIGH = _NEIGH


def neighbors(i):
    return NEIGH[i]


def obs_to_arrays(obs):
    """Convert observation lists to numpy-free Python lists (int arrays)."""
    vis = obs["visible"]
    own = obs["owners"]
    w = obs["weights"]
    c = obs["costs"]
    size = SIZE
    weight = [0] * N
    cost = [0] * N
    owner = [-3] * N  # -3 hidden
    for i in range(N):
        if vis[i]:
            owner[i] = own[i]
            if own[i] >= 0:
                weight[i] = w[i]
                cost[i] = c[i]
            elif own[i] == -1:
                weight[i] = w[i]
                cost[i] = c[i]
            elif own[i] == -2:
                weight[i] = 0
                cost[i] = 0
    return weight, cost, owner


class Board:
    """Determinized full board for planning."""
    def __init__(self, weight, cost, owner, energy, score, current, steps, rng=None):
        self.weight = weight[:]
        self.cost = cost[:]
        self.owner = owner[:]
        self.energy = list(energy)
        self.score = list(score)
        self.current = current
        self.steps = steps
        self.rng = rng or random.Random()

    # ---- determinization ----
    @staticmethod
    def from_obs(obs, rng=None, wall_p=0.19):
        rng = rng or random.Random()
        seat = obs["seat"]
        weight, cost, owner = obs_to_arrays(obs)
        # sample hidden cells
        for i in range(N):
            if owner[i] == -3:
                if rng.random() < wall_p:
                    owner[i] = -2
                    weight[i] = 0
                    cost[i] = 0
                else:
                    owner[i] = -1
                    weight[i] = rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
                    cost[i] = rng.choice([1, 2, 3, 4])
        # ensure start tiles
        owner[0] = 0
        owner[N - 1] = 1
        weight[0] = cost[0] = weight[N - 1] = cost[N - 1] = 0
        return Board(weight, cost, owner, list(obs["energy"]), list(obs["scores"]),
                     seat, 256 - obs["actions_remaining"], rng)

    # ---- move generation ----
    def legal_moves(self, p):
        """Return list of (cell, kind, cost, gain) plus (-1, 'rest', 0, 0) for None."""
        e = self.energy[p]
        moves = []
        # cells owned by p
        src = [i for i in range(N) if self.owner[i] == p]
        seen = set()
        for i in src:
            for j in NEIGH[i]:
                if j in seen:
                    continue
                o = self.owner[j]
                if o == -1:  # neutral
                    c = self.cost[j]
                    if c <= e:
                        seen.add(j)
                        moves.append((j, "claim", c, self.weight[j]))
                elif o == 1 - p:  # capture
                    c = self.cost[j] + 2
                    if c <= e:
                        seen.add(j)
                        moves.append((j, "capture", c, 2 * self.weight[j]))
        moves.append((-1, "rest", 0, 0))
        return moves

    def apply(self, p, action):
        cell, kind, cost_e, gain = action
        if kind == "rest":
            self.energy[p] = min(9, self.energy[p] + 3)
        else:
            self.energy[p] -= cost_e
            if kind == "claim":
                self.owner[cell] = p
                self.score[p] += self.weight[cell]
            elif kind == "capture":
                self.owner[cell] = p
                self.score[p] += self.weight[cell]
                self.score[1 - p] -= self.weight[cell]
        self.steps += 1
        self.current = 1 - p

    def margin(self, me):
        return self.score[me] - self.score[1 - me]