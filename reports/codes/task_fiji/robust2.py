"""Robustness batch 2: harder regimes + prior variants."""
import sys, math, random, statistics
sys.path.insert(0, ".")
import bench2
from bench2 import true_means
from reef import Ocean, Spot, Day, expected_catch
from candidates import MixUCB

def make_ocean(seed, regime, n_days=30, seed0=500):
    rng=random.Random(seed*7+13)
    if regime=="overlap":       # big values closer to steady means
        n_school=rng.choice([1,2,3])
        school=dict(p=rng.uniform(0.05,0.6), big=rng.uniform(5,11), big_sd=rng.uniform(1,3),
                    base=rng.uniform(0.2,1.0), base_sd=rng.uniform(0.2,0.5))
        steady=(1.0,4.0,0.5,1.5)
    elif regime=="many_school":
        n_school=rng.choice([4,5])
        school=dict(p=rng.uniform(0.02,0.3), big=rng.uniform(14,26), big_sd=rng.uniform(3,7),
                    base=rng.uniform(0.2,1.0), base_sd=rng.uniform(0.2,0.5))
        steady=(0.5,3.0,0.4,1.5)
    elif regime=="huge":        # big rewards much larger
        n_school=rng.choice([1,2])
        school=dict(p=rng.uniform(0.03,0.4), big=rng.uniform(35,60), big_sd=rng.uniform(5,10),
                    base=rng.uniform(0.2,1.0), base_sd=rng.uniform(0.2,0.5))
        steady=(0.5,3.0,0.5,1.5)
    elif regime=="practice_like":
        n_school=3
        school=dict(p=0.35, big=20.0, big_sd=5.0, base=0.5, base_sd=0.4)
        steady=(0.8,2.6,0.8,1.6)
    else:
        raise ValueError(regime)
    spots=[]
    for j in range(10):
        if j<n_school:
            spots.append(Spot(kind="school",
                p=max(0.01,school["p"]*rng.uniform(0.3,1.5)), big=school["big"]*rng.uniform(0.8,1.2),
                big_sd=school["big_sd"], base=school["base"], base_sd=school["base_sd"]))
        else:
            lo,hi,slo,shi=steady
            spots.append(Spot(kind="steady", mean=rng.uniform(lo,hi), sd=rng.uniform(slo,shi)))
    rng.shuffle(spots)
    # ensure a dominant arm exists (else threshold impossible); scale if best mean < 5.5
    seeds=tuple(seed0+seed*1000+i for i in range(n_days))
    return Ocean(n_spots=10, n_casts=1000, seeds=seeds, spots=tuple(spots), threshold=5300)

def evaluate(ctor, ocean):
    bench2.TRUE=true_means(ocean)
    from bench2 import best_label
    tot=[]; bc=[]
    for s in ocean.seeds:
        d=Day(ocean,s); ag=ctor(10,1000); bl=best_label(ocean,s); c=0
        for _ in range(ocean.n_casts):
            a=int(ag.choose()); x=d.cast(a); ag.update(a,x)
            if a==bl: c+=1
        tot.append(d.total); bc.append(c)
    ot=[]
    for s in ocean.seeds:
        d=Day(ocean,s); b=best_label(ocean,s); t=0.0
        for _ in range(ocean.n_casts): t+=d.cast(b)
        ot.append(t)
    return statistics.mean(tot), statistics.mean(ot), min(tot), min(bc), statistics.mean(bc)

if __name__=="__main__":
    cands={
      "base c.8 e280": lambda k,n: MixUCB(k,n,0.8,280),
      "prior a.5 b1.5": lambda k,n: MixUCB(k,n,0.8,280,a0=0.5,b0=1.5),
      "prior a1 b2": lambda k,n: MixUCB(k,n,0.8,280,a0=1.0,b0=2.0),
      "prior a.5 b.5": lambda k,n: MixUCB(k,n,0.8,280,a0=0.5,b0=0.5),
      "gap thr": lambda k,n: MixUCB(k,n,0.8,280,thr_mode="gap"),
      "c1.2 e280": lambda k,n: MixUCB(k,n,1.2,280),
    }
    regimes=["practice_like","overlap","many_school","huge"]
    rows={n:[] for n in cands}
    for regime in regimes:
        for rep in range(3):
            ocean=make_ocean(rep,regime)
            means=[expected_catch(s,6000) for s in ocean.spots]
            om=max(means)
            print(f"{regime:13s} rep{rep}: best={om:.1f} 2nd={sorted(means)[-2]:.1f} "
                  f"schools={sum(1 for s in ocean.spots if s.kind=='school')}")
            for nm,ctor in cands.items():
                random.seed(0)
                avg,o,mn,mnbc,mbc=evaluate(ctor,ocean)
                rows[nm].append((avg,o))
    print()
    for nm,vals in rows.items():
        ratios=[a/o for a,o in vals]
        print(f"{nm:16s} mean%oracle={100*statistics.mean(ratios):5.1f} min={100*min(ratios):5.1f} "
              f"per-case={[round(100*r,1) for r in ratios]}")
