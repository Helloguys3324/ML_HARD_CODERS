"""Benchmark harness for Island Duel agents against the native engine."""
import sys, time, random, json
from pathlib import Path

DATA = Path(__file__).parent / "data" / "island-duel-participant"
sys.path.insert(0, str(DATA))
from client import ArenaClient


def play_game(env, seat, choose):
    """Play one game from current reset state. Returns (seat, scores, winner, npc_turns)."""
    r = env.observe() if False else None
    raise NotImplementedError


def run_game(arena, seed, choose):
    with ArenaClient() as env:
        r = env.reset(arena, seed)
        o = r["observation"]
        seat = o["seat"]
        while not r["done"]:
            a = choose(o)
            r = env.step(a)
            o = r["observation"]
        return seat, r["scores"], r["winner"]


def benchmark(choose, arenas=None, seeds_per=None, seeds_list=None, verbose=True):
    """Run games and report win/draw/loss and avg score margin."""
    from collections import Counter
    wins = draws = losses = 0
    margins = []
    t0 = time.time()
    n = 0
    if arenas is None:
        arenas = range(32)
    if seeds_list is not None:
        for arena, seed in seeds_list:
            seat, scores, winner = run_game(arena, seed, choose)
            n += 1
            if winner is None:
                draws += 1
            elif winner == seat:
                wins += 1
            else:
                losses += 1
            margins.append(scores[seat] - scores[1 - seat])
    else:
        for arena in arenas:
            for seed in (seeds_per if seeds_per is not None else []):
                seat, scores, winner = run_game(arena, seed, choose)
                n += 1
                if winner is None:
                    draws += 1
                elif winner == seat:
                    wins += 1
                else:
                    losses += 1
                margins.append(scores[seat] - scores[1 - seat])
    t1 = time.time()
    if margins:
        avg_margin = sum(margins) / len(margins)
        pos = sum(1 for m in margins if m > 0) / len(margins)
    else:
        avg_margin = pos = 0.0
    res = {
        "games": n,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "winrate": (wins + 0.5 * draws) / n if n else 0,
        "avg_margin": avg_margin,
        "elapsed": t1 - t0,
    }
    if verbose:
        print(json.dumps(res))
    return res


if __name__ == "__main__":
    # quick self-test
    def naive(o):
        la = o["legal_actions"]
        return la[0]
    benchmark(naive, seeds_list=[(0, 144) for _ in range(1)])