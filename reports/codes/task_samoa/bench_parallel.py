"""Parallel benchmark using multiple engine processes."""
import sys, time, random, json
from pathlib import Path
from multiprocessing import Pool
import importlib
import importlib.util

DATA = Path(__file__).parent / "data" / "island-duel-participant"
sys.path.insert(0, str(DATA))
from client import ArenaClient

_HERE = Path(__file__).parent


def _make_choose(spec):
    # spec is (module_fname, factory_name, kwargs)
    mod_fname, factory_name, kwargs = spec
    sys.path.insert(0, str(mod_fname.parent))
    specobj = importlib.util.spec_from_file_location("agentmod", mod_fname)
    mod = importlib.util.module_from_spec(specobj)
    specobj.loader.exec_module(mod)
    factory = getattr(mod, factory_name)
    return factory(**kwargs) if kwargs else factory()


def run_one(task):
    arena, seed, spec = task
    choose = _make_choose(spec)
    with ArenaClient() as env:
        r = env.reset(arena, seed)
        o = r["observation"]
        seat = o["seat"]
        while not r["done"]:
            a = choose(o)
            r = env.step(a)
            o = r["observation"]
        return seat, r["scores"], r["winner"]


def benchmark_parallel(spec, seeds_list, workers=16):
    t0 = time.time()
    tasks = [(a, s, spec) for (a, s) in seeds_list]
    with Pool(workers) as p:
        results = p.map(run_one, tasks)
    wins = draws = losses = 0
    margins = []
    for seat, scores, winner in results:
        if winner is None:
            draws += 1
        elif winner == seat:
            wins += 1
        else:
            losses += 1
        margins.append(scores[seat] - scores[1 - seat])
    n = len(results)
    t1 = time.time()
    return {
        "games": n,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "winrate": (wins + 0.5 * draws) / n if n else 0,
        "avg_margin": sum(margins) / n if n else 0.0,
        "elapsed": t1 - t0,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arenas", type=int, default=32)
    args = p.parse_args()

    mod = _HERE / "agents_heuristic.py"
    agents = [
        ("greedy rho=1.0", (mod, "make_greedy", {"rest_rho": 1.0})),
        ("greedy rho=1.5", (mod, "make_greedy", {"rest_rho": 1.5})),
        ("greedy rho=2.0", (mod, "make_greedy", {"rest_rho": 2.0})),
        ("greedy rho=2.5", (mod, "make_greedy", {"rest_rho": 2.5})),
        ("greedy2 thr=0.8", (mod, "make_greedy2", {"claim_threshold": 0.8})),
        ("greedy2 thr=1.0", (mod, "make_greedy2", {"claim_threshold": 1.0})),
        ("greedy2 thr=1.2", (mod, "make_greedy2", {"claim_threshold": 1.2})),
        ("greedy3 rho=1.0", (mod, "make_greedy3", {"weight_rho": 1.0})),
        ("greedy3 rho=2.0", (mod, "make_greedy3", {"weight_rho": 2.0})),
    ]
    random.seed(args.seed)
    seeds_list = [(a, random.randint(0, 2**32 - 1)) for a in range(args.arenas) for _ in range(args.seeds)]
    for name, spec in agents:
        r = benchmark_parallel(spec, seeds_list, args.workers)
        print(f"{name:20s} winrate={r['winrate']:.3f} avg_margin={r['avg_margin']:.1f} ({r['wins']}W/{r['draws']}D/{r['losses']}L) {r['elapsed']:.1f}s")