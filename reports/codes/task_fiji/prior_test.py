"""Focused comparison: Beta prior variants on practice + targeted reefs."""
import sys, statistics
sys.path.insert(0, ".")
import bench2
from bench2 import true_means
from reef import Ocean, Day, expected_catch
from final_validate import make_targeted
from candidates import MixUCB
from main import Strategy

def play(ctor, ocean):
    bench2.TRUE = true_means(ocean)
    from bench2 import best_label
    tot=[]; mn=None
    for s in ocean.seeds:
        d=Day(ocean,s); ag=ctor(10,1000)
        for _ in range(1000):
            a=int(ag.choose()); x=d.cast(a); ag.update(a,x)
        tot.append(d.total)
    return statistics.mean(tot), min(tot)

if __name__=="__main__":
    cfg={
      "ship (a1,b1,c1)": lambda k,n: Strategy(k,n),
      "a2b1 c1.0": lambda k,n: MixUCB(k,n,1.0,300,a0=2.0,b0=1.0),
      "a2b1 c1.3": lambda k,n: MixUCB(k,n,1.3,300,a0=2.0,b0=1.0),
      "a3b1 c1.0": lambda k,n: MixUCB(k,n,1.0,300,a0=3.0,b0=1.0),
      "a1b1 c1.4": lambda k,n: MixUCB(k,n,1.4,300),
      "a2b2 c1.0": lambda k,n: MixUCB(k,n,1.0,300,a0=2.0,b0=2.0),
    }
    from reef import load_ocean
    practice=load_ocean("train-output.json")
    reefs=[make_targeted(rep) for rep in (1,3,5,7)]
    rows={n:[] for n in cfg}
    for nm,ctor in cfg.items():
        avg,mn=play(ctor,practice); rows[nm].append((avg,mn,"practice"))
        for i,oc in enumerate(reefs):
            avg,mn=play(ctor,oc); rows[nm].append((avg,mn,f"t{i}"))
    for nm,v in rows.items():
        print(f"{nm:18s} practice={v[0][0]:7.1f}  targets=" + " ".join(f"{a:7.1f}" for a,_,_ in v[1:])
              + f"  min_of_mins={min(m for _,m,_ in v):7.1f}")
