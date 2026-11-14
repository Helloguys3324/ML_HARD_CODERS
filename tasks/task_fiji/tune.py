"""Tuning MixUCB: c and exploration budget; plus confidence-based stop."""
import sys, math, random, statistics
sys.path.insert(0, ".")
from reef import load_ocean, Day, expected_catch
import bench2
from bench2 import MixBase, best_label, true_means

class MixUCB2(MixBase):
    def __init__(self, n_spots, n_casts, c=1.0, explode=320, stop_conf=None, eps=0.0):
        super().__init__(n_spots,n_casts)
        self.c=c; self.explode=explode; self.explo=False; self.stop_conf=stop_conf; self.eps=eps
    def choose(self):
        if self.explo:
            if self.eps and self.rng.random()<self.eps: return self.rng.randrange(self.k)
            return max(range(self.k), key=lambda i:self.mu[i])
        if self.t>=self.explode:
            self.explo=True; return max(range(self.k), key=lambda i:self.mu[i])
        if self.stop_conf is not None and self.t>30:
            order=sorted(range(self.k), key=lambda i:-self.mu[i])
            a,b=order[0],order[1]
            gap=self.mu[a]-self.mu[b]
            if gap > self.stop_conf*max(self.sd[b], 0.3):
                self.explo=True; return a
        best=-1e18; bi=0
        for i in range(self.k):
            if self.n[i]==0: v=1e9
            else: v=self.mu[i]+self.c*self.sd[i]
            if v>best: best=v; bi=i
        return bi

def run(name, ctor, ocean):
    tot=[]; bc_all=[]
    for seed in ocean.seeds:
        d=Day(ocean,seed); ag=ctor(ocean.n_spots,ocean.n_casts)
        bl=best_label(ocean,seed); bc=0
        for _ in range(ocean.n_casts):
            a=int(ag.choose()); x=d.cast(a); ag.update(a,x)
            if a==bl: bc+=1
        tot.append(d.total); bc_all.append(bc)
    print(f"{name:24s} avg={statistics.mean(tot):8.1f} min={min(tot):7.1f} "
          f"best={statistics.mean(bc_all):6.0f} worst_day_best={min(bc_all):4d}")

if __name__=="__main__":
    ocean=load_ocean("train-output.json")
    bench2.TRUE=true_means(ocean)
    tests=[]
    for c in (0.6,0.8,1.0,1.2):
        for e in (200,260,320,420):
            tests.append((f"c={c} e={e}", lambda k,n,c=c,e=e: MixUCB2(k,n,c,e)))
    for sc in (1.5,2.0,3.0):
        tests.append((f"c=1 conf{sc} e320", lambda k,n,sc=sc: MixUCB2(k,n,1.0,320,sc)))
        tests.append((f"c=1 conf{sc} e500", lambda k,n,sc=sc: MixUCB2(k,n,1.0,500,sc)))
    for nm,ct in tests:
        run(nm,ct,ocean)
