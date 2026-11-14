import numpy as np, glob, os
from scipy import ndimage

G='/home/kozak/mllm/MML/work/solutions/task_japan/gray'
ROOT='/home/kozak/mllm/MML/work/tasks/task_japan/data/dataset'

def load_cred(h, scale=1):
    ps=sorted(glob.glob(os.path.join(ROOT,'credited',h,'*.jpg')))
    ns=[os.path.relpath(p,ROOT).replace('/','__') for p in ps]
    A=np.stack([np.load(os.path.join(G,n+'.npy')).astype(np.float32)[20:700:scale,20:1260:scale] for n in ns])
    return A

def loo_corr(R):
    s=R.sum(0); n=len(R); K=(s[None]-R)/(n-1)
    out=[]
    for i in range(n):
        a=R[i].ravel(); b=K[i].ravel()
        a=a-a.mean(); b=b-b.mean()
        out.append(float((a*b).mean()/(a.std()*b.std()+1e-12)))
    return float(np.mean(out))

if __name__=='__main__':
    hs=sorted(os.listdir(os.path.join(ROOT,'credited')))
    # (sigma, downsample)
    scales=[(0.7,1),(1.2,1),(2,1),(4,2),(8,2),(16,4),(32,4),(64,8)]
    print('sigma      ' + '  '.join('%10s'%h for h in hs))
    for s,ds in scales:
        vals=[]
        for h in hs:
            Rg=load_cred(h,ds)
            R=Rg-ndimage.gaussian_filter(Rg,(0,s/ds,s/ds))
            vals.append(loo_corr(R))
        print('%6.1f     '%s + '  '.join('%+.5f'%v for v in vals), flush=True)
