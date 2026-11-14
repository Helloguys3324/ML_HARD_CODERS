"""Candidate strategies (final family) with configurable priors + threshold modes."""
import math, random

class MixUCB:
    def __init__(self, n_spots, n_casts, c=0.8, explode=280, a0=1.0, b0=1.0,
                 big_var=25.0, thr_mode="adapt", rng_seed=12345):
        self.k=n_spots; self.c=c; self.explode=explode
        self.a0=a0; self.b0=b0; self.big_var=big_var
        self.n=[0]*n_spots
        self.obs=[[] for _ in range(n_spots)]
        self.nb=[0]*n_spots; self.sb=[0.0]*n_spots
        self.ns=[0]*n_spots; self.ss=[0.0]*n_spots
        self.mu=[2.0]*n_spots; self.sd=[8.0]*n_spots
        self.thr=8.0; self.thr_mode=thr_mode
        self.t=0; self.explo=False
        self.rng=random.Random(rng_seed)
        self.checkpoints={15,30,60,120,240,480}
    def _refresh_thr(self):
        vals=[x for o in self.obs for x in o]
        if len(vals)>=25:
            vals.sort()
            if self.thr_mode=="adapt":
                new=0.5*(vals[len(vals)//2]+vals[-1])
            elif self.thr_mode=="gap":
                # largest gap in upper half of sorted values, threshold at its middle
                lo=len(vals)//2
                best_gap=0.0; new=vals[lo]
                for j in range(max(lo,1), len(vals)):
                    g=vals[j]-vals[j-1]
                    if g>best_gap:
                        best_gap=g; new=0.5*(vals[j]+vals[j-1])
                if best_gap < 0.35*vals[-1]:
                    new=0.5*(vals[len(vals)//2]+vals[-1])
            else:
                new=float(self.thr_mode)
            if new>0 and abs(new-self.thr)>0.2*self.thr:
                self.thr=new; self._rebuild()
    def _rebuild(self):
        for i in range(self.k):
            nb=ns=0; sb=ss=0.0
            for x in self.obs[i]:
                if x>=self.thr: nb+=1; sb+=x
                else: ns+=1; ss+=x
            self.nb[i]=nb; self.sb[i]=sb; self.ns[i]=ns; self.ss[i]=ss
        self._post()
    def _post(self):
        a0=self.a0; b0=self.b0
        for i in range(self.k):
            nb=self.nb[i]; ns=self.ns[i]
            if nb+ns==0: self.mu[i]=2.0; self.sd[i]=8.0; continue
            p=(a0+nb)/(a0+b0+nb+ns)
            mb=self.sb[i]/nb if nb else 20.0
            ms=self.ss[i]/ns if ns else 1.0
            self.mu[i]=p*mb+(1-p)*ms
            vp=p*(1-p)/(a0+b0+nb+ns+1.0)
            v=(mb-ms)**2*vp + p*p*self.big_var/max(1,nb) + (1-p)**2*0.3/max(1,ns)
            self.sd[i]=math.sqrt(max(v,1e-6))
    def choose(self):
        if self.explo: return max(range(self.k), key=lambda i:self.mu[i])
        if self.t>=self.explode:
            self.explo=True; return max(range(self.k), key=lambda i:self.mu[i])
        best=-1e18; bi=0
        for i in range(self.k):
            if self.n[i]==0: v=1e9
            else: v=self.mu[i]+self.c*self.sd[i]
            if v>best: best=v; bi=i
        return bi
    def update(self,i,x):
        self.t+=1; self.n[i]+=1; self.obs[i].append(x)
        if x>=self.thr: self.nb[i]+=1; self.sb[i]+=x
        else: self.ns[i]+=1; self.ss[i]+=x
        if self.t in self.checkpoints: self._refresh_thr()
        self._post()
