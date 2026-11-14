"""In-process benchmark of bandit strategies on the Fiji reef."""
import sys, math, random, statistics, json
sys.path.insert(0, ".")
from reef import load_ocean, play_all, Day, expected_catch

K = 10
N = 1000

# ---------------- strategies ----------------

class Greedy:
    def __init__(self, n_spots, n_casts):
        self.k = n_spots; self.n = [0]*n_spots; self.s = [0.0]*n_spots
    def choose(self):
        for i in range(self.k):
            if self.n[i] == 0: return i
        return max(range(self.k), key=lambda i: self.s[i]/self.n[i])
    def update(self, i, x):
        self.n[i] += 1; self.s[i] += x

class RandomArm:
    def __init__(self, n_spots, n_casts): self.k = n_spots
    def choose(self): return random.randrange(self.k)
    def update(self, i, x): pass

class UCB1:
    def __init__(self, n_spots, n_casts, c=1.0):
        self.k=n_spots; self.n=[0]*n_spots; self.s=[0.0]*n_spots; self.m2=[0.0]*n_spots
        self.c=c; self.t=0
    def choose(self):
        self.t += 1
        best=-1e18; bi=0
        lt=math.log(max(2,self.t))
        for i in range(self.k):
            if self.n[i]==0:
                v=1e17
            else:
                m=self.s[i]/self.n[i]
                v=m+self.c*math.sqrt(lt/self.n[i])
            if v>best: best=v; bi=i
        return bi
    def update(self,i,x):
        self.n[i]+=1; self.s[i]+=x

class TS:
    """Thompson sampling, normal posterior on the mean."""
    def __init__(self, n_spots, n_casts, prior_mu=2.5, prior_sd=8.0, inflate=1.0):
        self.k=n_spots; self.n=[0]*n_spots; self.s=[0.0]*n_spots; self.m2=[0.0]*n_spots
        self.pm=prior_mu; self.ps=prior_sd; self.inflate=inflate
        self.rng = random.Random(12345)
    def choose(self):
        best=-1e18; bi=0
        g=self.rng.gauss
        for i in range(self.k):
            n=self.n[i]
            if n==0:
                mu=g(self.pm,self.ps)
            else:
                m=self.s[i]/n
                if n>1:
                    var=self.m2[i]/(n-1)
                    sd=math.sqrt(var/n)*self.inflate
                else:
                    sd=self.ps
                mu=g(m,sd)
            if mu>best: best=mu; bi=i
        return bi
    def update(self,i,x):
        n=self.n[i]+1
        d=x-self.s[i]/self.n[i] if self.n[i]>0 else x-self.pm
        # standard Welford using running mean
        if self.n[i]==0:
            mean=x
        else:
            mean=self.s[i]/self.n[i]
        self.n[i]=n
        self.s[i]=mean*n
        # m2 update: use delta
        self.m2[i]+= (x-mean)*(x-mean) if self.n[i]==1 else 0.0
        # recompute m2 incrementally properly below

class TS2(TS):
    """TS with correct Welford m2."""
    def update(self,i,x):
        self.n[i]+=1; n=self.n[i]
        delta=x-(self.s[i]/max(1,n-1)) if n>1 else x
        # keep running mean in s[i]
        if n==1:
            self.s[i]=x; self.m2[i]=0.0
        else:
            mean=self.s[i]
            nm=mean+(x-mean)/n
            self.m2[i]+=(x-mean)*(x-nm)
            self.s[i]=nm
    def choose(self):
        best=-1e18; bi=0
        g=self.rng.gauss
        for i in range(self.k):
            n=self.n[i]
            if n==0:
                mu=g(self.pm,self.ps)
            else:
                m=self.s[i]
                if n>1:
                    var=self.m2[i]/(n-1)
                    sd=math.sqrt(max(var,1e-9)/n)*self.inflate
                else:
                    sd=self.ps
                mu=g(m,sd)
            if mu>best: best=mu; bi=i
        return bi

class WarmThenBest:
    """Each arm K times, then greedy best (with eps random exploration)."""
    def __init__(self, n_spots, n_casts, K=10, eps=0.0):
        self.k=n_spots; self.n=[0]*n_spots; self.s=[0.0]*n_spots
        self.K=K; self.eps=eps; self.t=0; self.rng=random.Random(7)
    def choose(self):
        self.t+=1
        for i in range(self.k):
            if self.n[i]<self.K: return i
        if self.eps and self.rng.random()<self.eps: return self.rng.randrange(self.k)
        return max(range(self.k), key=lambda i: self.s[i]/self.n[i])
    def update(self,i,x): self.n[i]+=1; self.s[i]+=x

class Racing:
    """Warm up all arms, then successive elimination; final exploit."""
    def __init__(self, n_spots, n_casts, M=4, margin=1.2, eps=0.03, start=3):
        self.k=n_spots; self.n=[0]*n_spots; self.s=[0.0]*n_spots; self.m2=[0.0]*n_spots
        self.M=M; self.margin=margin; self.eps=eps
        self.t=0; self.rng=random.Random(11)
        self.surv=list(range(n_spots)); self.round=0; self.done=False
        self.start=start
        self.phase_n=0
    def mean(self,i): return self.s[i]/self.n[i] if self.n[i] else 0.0
    def choose(self):
        self.t+=1
        if not self.done:
            # round-robin within current survivors
            if self.phase_n < len(self.surv)*self.M:
                i = self.surv[self.phase_n % len(self.surv)]
                self.phase_n += 1
                return i
            # end of round: eliminate
            self._eliminate()
            if self.done:
                pass
            else:
                self.phase_n = 0
                i = self.surv[self.phase_n % len(self.surv)]; self.phase_n += 1
                return i
        # exploit leader, eps explore
        if self.eps and self.rng.random()<self.eps:
            return self.rng.randrange(self.k)
        return max(range(self.k), key=self.mean)
    def _eliminate(self):
        k=self.k
        mu=[self.mean(i) for i in range(k)]
        sem=[math.sqrt(max(self.m2[i]/(self.n[i]-1),0)/self.n[i]) if self.n[i]>1 else 1e9 for i in range(k)]
        order=sorted(range(k), key=lambda i: -mu[i])
        top=order[0]
        keep=[]
        for i in order:
            if mu[i] + self.margin*sem[i] >= mu[top] - self.margin*sem[top]:
                keep.append(i)
            else:
                break
        # never shrink below 2 survivors unless budget low
        if len(keep)<2: keep=order[:2]
        self.surv=keep
        self.M = max(self.M, 6)
        if len(keep)<=1 or self.t > 700:
            self.done=True
    def update(self,i,x):
        self.n[i]+=1; n=self.n[i]
        if n==1: self.s[i]=x; self.m2[i]=0.0
        else:
            m=self.s[i]; nm=m+(x-m)/n; self.m2[i]+=(x-m)*(x-nm); self.s[i]=nm

class MixTS:
    """TS with mixture-aware mean estimate: split catches at threshold big/small."""
    def __init__(self, n_spots, n_casts, thr=8.0, prior_sd=8.0):
        self.k=n_spots; self.n=[0]*n_spots; self.s=[0.0]*n_spots
        self.nb=[0]*n_spots; self.sb=[0.0]*n_spots  # big counts/sums
        self.ns=[0]*n_spots; self.ss=[0.0]*n_spots  # small
        self.thr=thr; self.ps=prior_sd; self.rng=random.Random(3)
        self.pm=2.0
    def choose(self):
        best=-1e18; bi=0; g=self.rng.gauss
        for i in range(self.k):
            n=self.n[i]
            if n==0:
                mu=g(self.pm,self.ps)
            else:
                # posterior on p (big) with Beta(1,1), and on big/small means
                a=1+self.nb[i]; b=1+self.ns[i]
                # sample p via two gammas
                p=self._bg(a,b)
                if self.nb[i]>0:
                    mb=self.sb[i]/self.nb[i]
                    sdb=max(2.0, 5.0/math.sqrt(self.nb[i]))
                else:
                    mb=20.0; sdb=5.0
                if self.ns[i]>0:
                    ms=self.ss[i]/self.ns[i]
                    sds=max(0.3, 1.0/math.sqrt(self.ns[i]))
                else:
                    ms=1.0; sds=1.0
                mu=p*g(mb,sdb)+(1-p)*g(ms,sds)
            if mu>best: best=mu; bi=i
        return bi
    def _bg(self,a,b):
        # sample Beta(a,b) via ratio of gammas (stdlib has no gamma; use gammavariate)
        x=self.rng.gammavariate(a,1.0); y=self.rng.gammavariate(b,1.0)
        return x/(x+y)
    def update(self,i,x):
        self.n[i]+=1; self.s[i]+=x
        if x>=self.thr: self.nb[i]+=1; self.sb[i]+=x
        else: self.ns[i]+=1; self.ss[i]+=x

# ---------------- evaluation ----------------

def oracle(ocean):
    tot=[]
    for seed in ocean.seeds:
        order=list(range(ocean.n_spots)); random.Random(seed).shuffle(order)
        best=order.index(5) if ocean.n_spots==10 else 0
        d=Day(ocean,seed)
        t=0.0
        for _ in range(ocean.n_casts): t+=d.cast(best)
        tot.append(t)
    return tot

def report(name, totals, thresh):
    m=statistics.mean(totals)
    print(f"{name:22s} avg={m:8.1f}  min={min(totals):7.1f} max={max(totals):7.1f}  pass={'Y' if m>=thresh else 'N'}")

if __name__=="__main__":
    ocean=load_ocean("train-output.json")
    th=ocean.threshold
    print("spot true means:", [round(expected_catch(s,20000),2) for s in ocean.spots])
    ot=oracle(ocean); 
    print(f"ORACLE (play best spot) avg={statistics.mean(ot):.1f}")
    tests=[
        ("greedy", Greedy),
        ("random", RandomArm),
        ("ucb1 c=1", lambda k,n: UCB1(k,n,1.0)),
        ("ucb1 c=2", lambda k,n: UCB1(k,n,2.0)),
        ("ts", TS2),
        ("ts infl1.5", lambda k,n: TS2(k,n,inflate=1.5)),
        ("warm10", lambda k,n: WarmThenBest(k,n,10,0.0)),
        ("warm10 eps.02", lambda k,n: WarmThenBest(k,n,10,0.02)),
        ("warm20 eps.02", lambda k,n: WarmThenBest(k,n,20,0.02)),
        ("racing", Racing),
        ("mixts", MixTS),
    ]
    for name,ctor in tests:
        random.seed(0)
        tot=play_all(ctor,ocean)
        report(name,tot,th)
