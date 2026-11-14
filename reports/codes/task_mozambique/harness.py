"""Local harness: play agent() against the provided engine, in-process.

Usage:
    from harness import Arena, play
    res = play(agent_fn, group, seed, game)
"""
import json
import os
import sys
import time

DATA = '/home/kozak/mllm/MML/work/tasks/task_mozambique/data'
sys.path.insert(0, DATA)
from arena import Arena  # noqa: E402


def play(agent, group, seed, game, trace=False):
    """agent(observation, memory) -> action.  Returns dict with result etc."""
    mem = {}
    timeline = []
    with Arena() as ar:
        st = ar.reset(group, seed, game)
        prev = None
        n = 0
        while not st['game_done']:
            obs = st['observation']
            if obs['round'] == 0:
                mem = {}
            t0 = time.monotonic()
            act = agent(obs, mem)
            dt = time.monotonic() - t0
            if prev is not None:
                timeline.append((obs['round'], obs['own']['score'], obs['opponent_score'], obs['claims'][:], dt))
            st = ar.step(act)
            prev = obs
            n += 1
            if n > 100:
                raise RuntimeError('too many steps')
        res = st['result']
        o = st['observation']
        if trace:
            return dict(group=group, seed=seed, game=game, scores=res['scores'],
                        half=res['half_points'], claims=o['claims'], rounds=n, timeline=timeline)
        return dict(group=group, seed=seed, game=game, scores=res['scores'],
                    half=res['half_points'], claims=o['claims'], rounds=n)


if __name__ == '__main__':
    # quick smoke: starter agent
    import importlib.util
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA, 'main.py')
    spec = importlib.util.spec_from_file_location('m', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    print(play(m.agent, 0, 767302775, 0, trace=True)['scores'])
