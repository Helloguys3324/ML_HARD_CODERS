#!/usr/bin/env python3
"""Reproducible local evaluation for the Mozambique task.

Steps (already done once in this workspace):
  1. fetch the participant zip:  ../py tools/fetch_drive_direct.py --task-id task_mozambique
  2. unzip mozambique-participant.zip into work/tasks/task_mozambique/data
  3. NixOS only: patch the engine interpreter (generic Linux glibc):
       nix-shell -p patchelf --run 'patchelf --set-interpreter /nix/store/...glibc.../lib/ld-linux-x86-64.so.2 \
            --set-rpath /nix/store/...glibc.../lib mozambique-engine'
  4. run:  /home/kozak/mllm/py solve.py [maps] [seed] [jobs]

`solve.py` plays `main.py` exactly like the evaluator (in-process agent function,
same Arena SDK) on `maps` random practice maps x 5 games and reports the
worst-game-per-map match score.
"""
import importlib.util
import json
import random
import sys
import time
from multiprocessing import Pool

HERE = '/home/kozak/mllm/MML/work/solutions/task_mozambique'
DATA = '/home/kozak/mllm/MML/work/tasks/task_mozambique/data'
sys.path.insert(0, HERE)
from harness import play  # noqa: E402

AGENT_PATH = HERE + '/main.py'


def load_agent(path=AGENT_PATH):
    spec = importlib.util.spec_from_file_location('submitted_agent', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.agent


def one(args):
    index, group, seed, game = args
    try:
        r = play(load_agent(), group, seed, game)
        return (index, game, r['half'], r['scores'], r['claims'])
    except Exception as e:  # a crash in the real evaluator means 0 points
        return (index, game, -1, repr(e), None)


def main():
    maps = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    rng = random.Random(seed)
    games = [(m, m % 8, rng.randrange(2 ** 32), p) for m in range(maps) for p in range(5)]
    t0 = time.time()
    with Pool(jobs) as pool:
        out = pool.map(one, games, chunksize=1)
    by_map = {}
    for index, game, half, scores, claims in out:
        by_map.setdefault(index, {})[game] = half
    worst = [min(v.values()) for v in by_map.values()]
    report = dict(maps=maps, seed=seed,
                  map_wins=sum(w == 2 for w in worst),
                  map_draws=sum(w == 1 for w in worst),
                  map_losses=sum(w == 0 for w in worst),
                  match_score=round(sum(worst) / (2 * maps), 4),
                  points=sum(worst), out_of=2 * maps,
                  seconds=round(time.time() - t0, 1))
    print(json.dumps(report))
    for index, game, half, scores, claims in sorted(out):
        if half != 2:
            print('game', index, game, 'half', half, 'scores', scores, 'claims', claims)


if __name__ == '__main__':
    main()
