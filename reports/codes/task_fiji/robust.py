"""Robustness: evaluate candidate strategies on synthetic reefs with varied parameters."""
import sys, math, random, statistics
sys.path.insert(0, ".")
import bench2
from bench2 import MixBase, best_label, true_means
from reef import Ocean, Spot, Day, expected_catch
from tune import MixUCB2, run
from bench2 import MixGreedy
from bench import UCB1

def make_ocean(seed, n_school=None, n_days=40, seed0=1000):
    rng=random.Random(seed)
    if n_school is None: n_school=rng.choice([0,1,1,2,2,3])
    k=10
    spots=[]
    idx=list(range(k)); rng.shuffle(idx)
    for j in range(k):
        if j < n_school:
            spots.append(Spot(kind="school", p=rng.uniform(0.02,0.45),
                              big=rng.uniform(12,30), big_sd=rng.uniform(2,8),
                              base=rng.uniform(0.2,1.2), base_sd=rng.uniform(0.2,0.6)))
        else:
            spots.append(Spot(kind="steady", mean=rng.uniform(0.5,3.5), sd=rng.uniform(0.4,1.8)))
    seeds=tuple(seed0+seed*1000+i for i in range(n_days))
    return Ocean(n_spots=k, n_casts=1000, seeds=seeds, spots=tuple(spots), threshold=5300)

def evaluate(ctor, ocean):
    bench2.TRUE=true_means(ocean)
    tot=[]; bc=[]
    for s in ocean.seeds:
        d=Day(ocean,s); ag=ctor(ocean.n_spots,ocean.n_casts)
        bl=best_label(ocean,s); c=0
        for _ in range(ocean.n_casts):
            a=int(ag.choose()); x=d.cast(a); ag.update(a,x)
            if a==bl: c+=1
        tot.append(d.total); bc.append(c)
    # oracle
    ot=[]
    for s in ocean.seeds:
        d=Day(ocean,s); b=best_label(ocean,s); t=0.0
        for _ in range(ocean.n_casts): t+=d.cast(b)
        ot.append(t)
    om=statistics.mean(ot)
    return statistics.mean(tot), om, min(tot), min(bc), statistics.mean(bc)

if __name__=="__main__":
    cands={
      "MixUCB c.8 e260": lambda k,n: MixUCB2(k,n,0.8,260),
      "MixUCB c.6 e260": lambda k,n: MixUCB2(k,n,0.6,260),
      "MixUCB c1 e320": lambda k,n: MixUCB2(k,n,1.0,320),
      "MixGreedy w10": lambda k,n: MixGreedy(k,n,10),
      "UCB1 c2": lambda k,n: UCB1(k,n,2.0),
    }
    rows={n:[] for n in cands}
    for reef in range(8):
        ocean=make_ocean(reef)
        means=[round(expected_catch(s,8000),2) for s in ocean.spots]
        nsch=sum(1 for s in ocean.spots if s.kind=="school")
        for nm,ctor in cands.items():
            random.seed(0)
            avg,om,mn,mnbc,mbc=evaluate(ctor,ocean)
            rows[nm].append((avg/om, avg, om, mn, mnbc))
        print(f"reef {reef}: schools={nsch} means={means}")
    print()
    for nm,vals in rows.items():
        ratios=[v[0] for v in vals]
        print(f"{nm:18s} mean%oracle={100*statistics.mean(ratios):5.1f} min%oracle={100*min(ratios):5.1f} "
              f"avg_best_casts={statistics.mean([v[1] for v in vals]):.0f}")
        print("   per-reef ratios:", [round(100*r,1) for r in ratios])
