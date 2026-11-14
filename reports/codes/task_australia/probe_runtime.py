import sys, time, traceback, gc
import numpy as np
import lab, lab.level as L, lab.engine as E, lab.brain as B, lab.fitness as F, lab.evolution as V
out=[]
def log(s):
    out.append(str(s)); open('/tmp/probe_out.txt','w').write('\n'.join(out)+'\n')

# pause the main App so it doesn't race on module globals
try:
    import __main__ as M
    apps=[o for o in gc.get_objects() if isinstance(o, M.App)]
    log('apps found %d' % len(apps))
    for a in apps:
        try: a.playing=False
        except Exception: pass
except Exception:
    log('pause err\n'+traceback.format_exc())

def run_brain(b, max_steps=6000):
    env=E.MarathonEnv(); env.reset()
    done=False; steps=0; total=0.0; coins=[]; info={}; events=[]
    while not done and steps<max_steps:
        act=b.act(env)
        ob,rew,done,info=env.step(act)
        r=F.reward(info); total+=r; steps+=1
        for c in info.get('coins_taken',[]):
            if c not in coins:
                coins.append(c); events.append((c,round(float(r),2),round(float(total),2)))
    return dict(steps=steps,total=total,coins=coins,info=info,events=events)

try:
    g=np.load('lab/proven.npy'); b=B.Brain.from_genome(g)
    for i in range(3):
        r=run_brain(b)
        log('proven run %d: steps=%d fitness=%.2f won=%r max_x=%.2f coins=%r' % (i,r['steps'],r['total'],r['info'].get('won'),r['info'].get('max_x'),r['coins']))
except Exception:
    log('PROVEN ERR\n'+traceback.format_exc())

try:
    lab=V.Lab(24,1,0.07)
    log('lab.proven %r' % (lab.proven,))
    if getattr(lab,'proven',None) is not None:
        try:
            gp=lab.proven.genome(); log('proven in lab genome diff to npy %r' % (float(np.abs(gp-g).max()),))
        except Exception: log('proven genome err '+traceback.format_exc())
except Exception:
    log('LAB INIT ERR\n'+traceback.format_exc())

# fast env for evolution
try:
    _orig=E.MarathonEnv
    class FastEnv(_orig):
        def __init__(self, *a, **k):
            k['max_steps']=1600
            _orig.__init__(self,*a,**k)
    E.MarathonEnv=FastEnv
    lab=V.Lab(24,1,0.07)
    t0=time.time()
    for gen in range(1,16):
        nsteps=0
        while not lab.finished:
            lab.step(); nsteps+=1
            if nsteps>5000: break
        led=lab.leader()
        log('GEN %d done t=%.1f nsteps=%d best_score=%.2f' % (gen,time.time()-t0,nsteps,led.score))
        # save genome + evaluate
        try:
            gg=led.brain.genome()
            rr=run_brain(led.brain, max_steps=1600)
            log('  gen%d champ run: steps=%d fit=%.2f won=%r max_x=%.2f coins=%r' % (gen,rr['steps'],rr['total'],rr['info'].get('won'),rr['info'].get('max_x'),rr['coins']))
        except Exception:
            log('  champ eval err '+traceback.format_exc())
        lab.finish()
    log('evolve done t=%.1f' % (time.time()-t0))
except Exception:
    log('EVOLVE ERR\n'+traceback.format_exc())
