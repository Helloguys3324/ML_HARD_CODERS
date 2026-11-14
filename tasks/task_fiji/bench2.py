"""Benchmark 2: mixture-aware strategies + diagnostics (fraction of casts on best)."""
import sys, math, random, statistics
sys.path.insert(0, ".")
from reef import load_ocean, Day, expected_catch

def true_means(ocean):
    return [expected_catch(s, 20000) for s in ocean.spots]

TRUE = []

def best_label(ocean, seed):
    order=list(range(ocean.n_spots)); random.Random(seed).shuffle(order)
    b=max(range(ocean.n_spots), key=lambda j: TRUE[j])
    return order.index(b)

class MixBase:
    """Mixture model: adaptive threshold + Beta-Bernoulli big/small components."""
    def __init__(self, n_spots, n_casts):
        self.k=n_spots
        self.n=[0]*n_spots
        self.obs=[[] for _ in range(n_spots)]
        self.nb=[0]*n_spots; self.sb=[0.0]*n_spots
        self.ns=[0]*n_spots; self.ss=[0.0]*n_spots
        self.mu=[2.0]*n_spots; self.sd=[8.0]*n_spots
        self.thr=8.0
        self.t=0
        self.rng=random.Random(99)
        self.checkpoints={15, 30, 60, 120, 240, 480}
        self.last_thr=self.thr
    def refresh_threshold(self):
        vals=[x for o in self.obs for x in o]
        if len(vals)>=25:
            vals.sort()
            new=0.5*(vals[len(vals)//2]+vals[-1])
            if abs(new-self.thr)>0.25*self.thr:
                self.thr=new
                self._rebuild()
    def _rebuild(self):
        for i in range(self.k):
            nb=ns=0; sb=ss=0.0
            for x in self.obs[i]:
                if x>=self.thr: nb+=1; sb+=x
                else: ns+=1; ss+=x
            self.nb[i]=nb; self.sb[i]=sb; self.ns[i]=ns; self.ss[i]=ss
        self._post()
    def _post(self):
        for i in range(self.k):
            nb=self.nb[i]; ns=self.ns[i]
            if nb+ns==0:
                self.mu[i]=2.0; self.sd[i]=8.0; continue
            p=(1.0+nb)/(2.0+nb+ns)
            mb=self.sb[i]/nb if nb else 20.0
            ms=self.ss[i]/ns if ns else 1.0
            self.mu[i]=p*mb+(1-p)*ms
            vp=p*(1-p)/(3.0+nb+ns)
            v=(mb-ms)**2*vp + p*p*25.0/max(1,nb) + (1-p)**2*0.3/max(1,ns)
            self.sd[i]=math.sqrt(max(v,1e-6))
    def update(self,i,x):
        self.t+=1
        self.n[i]+=1; self.obs[i].append(x)
        if x>=self.thr: self.nb[i]+=1; self.sb[i]+=x
        else: self.ns[i]+=1; self.ss[i]+=x
        if self.t in self.checkpoints:
            self.refresh_threshold()
        self._post()

class MixTS(MixBase):
    def __init__(self, n_spots, n_casts, explode=320, c=1.0):
        super().__init__(n_spots,n_casts)
        self.explode=explode; self.explo=False; self.c=c
    def choose(self):
        if self.explo: return max(range(self.k), key=lambda i:self.mu[i])
        if self.t>=self.explode:
            self.explo=True
            return max(range(self.k), key=lambda i:self.mu[i])
        g=self.rng.gauss; best=-1e18; bi=0
        for i in range(self.k):
            if self.n[i]==0: mu=g(2.0,8.0)
            else: mu=self.mu[i]+self.c*self.sd[i]*g(0,1)
            if mu>best: best=mu; bi=i
        return bi

class MixUCB(MixBase):
    def __init__(self, n_spots, n_casts, c=1.0, explode=320):
        super().__init__(n_spots,n_casts)
        self.c=c; self.explode=explode; self.explo=False
    def choose(self):
        if self.explo: return max(range(self.k), key=lambda i:self.mu[i])
        if self.t>=self.explode:
            self.explo=True
            return max(range(self.k), key=lambda i:self.mu[i])
        best=-1e18; bi=0
        for i in range(self.k):
            if self.n[i]==0: v=1e9
            else: v=self.mu[i]+self.c*self.sd[i]
            if v>best: best=v; bi=i
        return bi

class MixGreedy(MixBase):
    def __init__(self, n_spots, n_casts, warm=10, eps=0.0, switch=0, recency=0):
        super().__init__(n_spots,n_casts)
        self.warm=warm; self.eps=eps; self.switch=switch; self.recency=recency
    def choose(self):
        for i in range(self.k):
            if self.n[i]<self.warm: return i
        if self.eps and self.rng.random()<self.eps: return self.rng.randrange(self.k)
        return max(range(self.k), key=lambda i:self.mu[i])

def run(name, ctor, ocean):
    tot=[]; best_cast=[]
    for seed in ocean.seeds:
        d=Day(ocean,seed); ag=ctor(ocean.n_spots,ocean.n_casts)
        bl=best_label(ocean,seed); bc=0
        for _ in range(ocean.n_casts):
            a=int(ag.choose()); x=d.cast(a); ag.update(a,x)
            if a==bl: bc+=1
        tot.append(d.total); best_cast.append(bc)
    print(f"{name:20s} avg={statistics.mean(tot):8.1f} min={min(tot):7.1f} best_casts={statistics.mean(best_cast):6.0f}/1000")

if __name__=="__main__":
    ocean=load_ocean("train-output.json")
    TRUE = true_means(ocean)
    globals()['TRUE']=TRUE
    ot=[]
    for s in ocean.seeds:
        d=Day(ocean,s); b=best_label(ocean,s); t=0.0
        for _ in range(ocean.n_casts): t+=d.cast(b)
        ot.append(t)
    print("true means", [round(x,2) for x in TRUE], " oracle avg", round(statistics.mean(ot),1))
    tests=[
        ("MixGreedy w10", lambda k,n: MixGreedy(k,n,10)),
        ("MixGreedy w10 e.02", lambda k,n: MixGreedy(k,n,10,0.02)),
        ("MixGreedy w15", lambda k,n: MixGreedy(k,n,15)),
        ("MixTS exp320", MixTS),
        ("MixTS exp400", lambda k,n: MixTS(k,n,400)),
        ("MixTS exp500", lambda k,n: MixTS(k,n,500)),
        ("MixTS exp700", lambda k,n: MixTS(k,n,700)),
        ("MixUCB c1 320", MixUCB),
        ("MixUCB c2 320", lambda k,n: MixUCB(k,n,2.0)),
        ("MixUCB c2 500", lambda k,n: MixUCB(k,n,2.0,500)),
    ]
    for nm,ct in tests:
        run(nm,ct,ocean)
