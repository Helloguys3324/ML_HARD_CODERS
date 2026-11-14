import importlib.util
import json
import random
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, '/home/kozak/mllm/MML/work/solutions/task_mozambique')
from harness import play

AGENT_PATH = '/home/kozak/mllm/MML/work/solutions/task_mozambique/agent_dev.py'


def load_agent(path=AGENT_PATH):
    spec = importlib.util.spec_from_file_location('a', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.agent


def one(args):
    m, group, seed, game = args
    agent = load_agent()
    try:
        r = play(agent, group, seed, game)
        return (m, game, r['half'], r['scores'], r['claims'])
    except Exception as e:
        return (m, game, -1, str(e), None)


if __name__ == '__main__':
    maps = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    seed0 = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    jobs_n = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    path = sys.argv[4] if len(sys.argv) > 4 else AGENT_PATH
    rng = random.Random(seed0)
    games = [(m, (m % 8), rng.randrange(2 ** 32), p) for m in range(maps) for p in range(5)]
    t0 = time.time()
    with Pool(jobs_n) as pool:
        out = pool.map(one, games, chunksize=1)
    by_map = {}
    for m, game, half, scores, claims in out:
        by_map.setdefault(m, {})[game] = half
    worst = [min(v.values()) for v in by_map.values()]
    print(json.dumps(dict(maps=maps,
                          map_wins=sum(w == 2 for w in worst),
                          map_draws=sum(w == 1 for w in worst),
                          map_losses=sum(w == 0 for w in worst),
                          match_score=round(sum(worst) / (2 * maps), 4),
                          points=sum(worst), out_of=2 * maps,
                          seconds=round(time.time() - t0, 1))))
    for m, game, half, scores, claims in sorted(out):
        if half != 2:
            print('bad', m, game, half, scores, claims)
