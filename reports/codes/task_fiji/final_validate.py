"""Final validation of the shipped main.py Strategy on targeted synthetic reefs."""
import sys, random, statistics
sys.path.insert(0, ".")
import bench2
from bench2 import true_means, best_label
from reef import Ocean, Spot, Day, expected_catch
from main import Strategy

def make_targeted(seed, n_days=30):
    """Practice-like reef: one clear best school arm (best mean 5.5-9)."""
    rng = random.Random(seed * 31 + 7)
    best = rng.uniform(5.5, 9.0)
    p = rng.uniform(0.25, 0.45)
    big = rng.uniform(18, 23)
    base = 0.6
    # solve p*big + (1-p)*base = best  -> adjust big
    big = (best - (1 - p) * base) / p
    spots = [Spot(kind="school", p=p, big=big, big_sd=rng.uniform(3.5, 6),
                  base=base, base_sd=0.4)]
    n_school2 = rng.choice([0, 0, 1, 2])
    for _ in range(n_school2):
        spots.append(Spot(kind="school", p=rng.uniform(0.04, 0.15),
                          big=rng.uniform(16, 22), big_sd=rng.uniform(4, 6),
                          base=0.6, base_sd=0.4))
    while len(spots) < 10:
        spots.append(Spot(kind="steady", mean=rng.uniform(0.8, 2.7),
                          sd=rng.uniform(0.8, 1.6)))
    rng.shuffle(spots)
    seeds = tuple(2000 + seed * 100 + i for i in range(n_days))
    return Ocean(n_spots=10, n_casts=1000, seeds=seeds, spots=tuple(spots), threshold=5300)

def play(ctor, ocean):
    bench2.TRUE = true_means(ocean)
    tot = []
    for s in ocean.seeds:
        d = Day(ocean, s); ag = ctor(10, 1000)
        for _ in range(1000):
            a = int(ag.choose()); x = d.cast(a); ag.update(a, x)
        tot.append(d.total)
    ot = []
    for s in ocean.seeds:
        d = Day(ocean, s); b = best_label(ocean, s); t = 0.0
        for _ in range(1000):
            t += d.cast(b)
        ot.append(t)
    return statistics.mean(tot), statistics.mean(ot), min(tot)

if __name__ == "__main__":
    scores = []
    for rep in range(10):
        oc = make_targeted(rep)
        m = [expected_catch(s, 8000) for s in oc.spots]
        avg, orc, mn = play(Strategy, oc)
        scores.append((avg, orc))
        print(f"rep{rep}: best={max(m):5.2f} 2nd={sorted(m)[-2]:5.2f} "
              f"-> avg={avg:7.1f} ({100*avg/orc:5.1f}% oracle) min_day={mn:7.1f}")
    print(f"\nmean = {statistics.mean(a for a,o in scores):.1f}  "
          f"worst = {min(a for a,o in scores):.1f}  "
          f"all_pass = {all(a>=5300 for a,o in scores)}")
