import numpy as np, traceback, gc, time
import lab, lab.level as L, lab.engine as E, lab.brain as B, lab.fitness as F, lab.evolution as V
out=[]
def log(s):
    out.append(str(s)); open('/tmp/probe_out.txt','w').write('\n'.join(out)+'\n')
try:
    import __main__ as M
    for a in [o for o in gc.get_objects() if isinstance(o, M.App)]:
        a.playing=False
except Exception: pass

def run(b):
    env=E.MarathonEnv(); env.reset(); done=False; steps=0; tot=0.0; events=[]; seen=[]
    while not done and steps<6000:
        ob,rew,done,info=env.step(b.act(env))
        r=F.reward(info); tot+=r; steps+=1
        for c in info.get('coins_taken',[]):
            if c not in seen:
                seen.append(c); events.append((c,round(float(r),3),round(float(tot),3)))
    return steps,tot,bool(info.get('won')),float(info.get('max_x')),seen,events

try:
    g=np.load('lab/proven.npy'); b=B.Brain.from_genome(g)
    s,t,w,mx,seen,ev=run(b)
    log('PROVEN steps=%d fitness=%.3f won=%r max_x=%.3f' % (s,t,w,mx))
    log('COINS %r' % (seen,))
    log('EVENTS (coin, step_reward, cumulative): %r' % (ev,))
    log('SIGNS %r' % ([(x,F._SIGN[(x,11)]) for x in L.COIN_SITES],))
    log('SPREAD(1.0)=%r SPREAD(-1.0)=%r SPREAD(0.5)=%r' % (F._spread(1.0),F._spread(-1.0),F._spread(0.5)))
except Exception:
    log('ERR\n'+traceback.format_exc())
try:
    lab=V.Lab(24,1,0.07)
    same=sum(1 for br in lab.population if np.array_equal(br.genome(),g))
    log('initial population size=%d exact-proven copies=%d rescues=%r stagnant=%r' % (len(lab.population),same,lab.rescues,lab.stagnant))
except Exception:
    log('LABERR\n'+traceback.format_exc())
