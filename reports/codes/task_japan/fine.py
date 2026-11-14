import numpy as np, glob, os, json, collections
from scipy import ndimage

C='/home/kozak/mllm/MML/work/solutions/task_japan/cache'
ROOT='/home/kozak/mllm/MML/work/tasks/task_japan/data/dataset'

def load(n):
    return np.load(os.path.join(C,n+'.npy')).astype(np.float32)

def feats(r):
    # r: residual crop
    out={}
    # phase means 2x2, 3x3, 4x4, 8x8
    for p in [2,3,4,8]:
        H,W=r.shape
        hh=H//p*p; ww=W//p*p
        rr=r[:hh,:ww].reshape(hh//p,p,ww//p,p)
        pm=rr.mean(axis=(0,2))  # p x p
        out['phase%d'%p]=pm.ravel().tolist()
    # autocorrelation at small lags
    a=r-r.mean()
    v=(a*a).mean()
    for dy,dx in [(0,1),(1,0),(1,1),(0,2),(2,0),(0,3),(3,0),(2,2),(0,8),(8,0)]:
        out['ac_%d_%d'%(dy,dx)]=float((a[dy:,dx:]*a[:a.shape[0]-dy or None,:a.shape[1]-dx or None]).mean()/v) if dy or dx else 1.0
    # histogram normalized
    out['skew']=float(((a/a.std())**3).mean())
    out['kurt']=float(((a/a.std())**4).mean())
    return out

def collect(names, prefix):
    F={}
    for i,n in enumerate(names):
        F[n]=feats(load(prefix+n))
    return F

def main():
    cred={}
    for h in sorted(os.listdir(os.path.join(ROOT,'credited'))):
        names=[os.path.basename(p) for p in sorted(glob.glob(os.path.join(ROOT,'credited',h,'*.jpg')))]
        cred[h]=collect(names, 'credited__%s__'%h)
    anon_names=[os.path.basename(p) for p in sorted(glob.glob(os.path.join(ROOT,'anonymous','*.jpg')))]
    anon=collect(anon_names,'anonymous__')
    json.dump({'credited':cred,'anon':anon}, open('/home/kozak/mllm/MML/work/solutions/task_japan/fine.json','w'))
    # summarize per person
    keys=list(next(iter(cred['yakin'].values())).keys())
    for k in keys:
        if k.startswith('phase'):
            continue
        line='%-8s'%k
        for h in cred:
            vals=[v[k] for v in cred[h].values()]
            line+=' %8.4f'%np.mean(vals)
        print(line)
    print('\nphase2 means:')
    for h in cred:
        arr=np.array([v['phase2'] for v in cred[h].values()])
        print('%-12s'%h, np.round(arr.mean(0),5), 'std', np.round(arr.std(0),5))
    print('\nphase4 means:')
    for h in cred:
        arr=np.array([v['phase4'] for v in cred[h].values()])
        print('%-12s'%h, np.round(arr.mean(0),5))

if __name__=='__main__':
    main()
