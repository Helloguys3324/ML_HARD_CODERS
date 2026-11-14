import numpy as np, glob, os
from PIL import Image

CACHE='/home/kozak/mllm/MML/work/solutions/task_japan/cache'

def names_for(h):
    return sorted(os.path.basename(p) for p in glob.glob(f'/home/kozak/mllm/MML/work/tasks/task_japan/data/dataset/credited/{h}/*.jpg'))

def main():
    for h in ['toorisugari','yakin','amayadori']:
        ns=names_for(h)
        R=np.stack([np.load(f'{CACHE}/credited__{h}__{n}.npy').astype(np.float32) for n in ns])
        print(h, R.shape, 'per-image std mean %.2f'%R.std(axis=(1,2)).mean())
        K=R.mean(0)
        print('  K std %.2f  (expected if pattern std %.2f: %.2f)'%(K.std(), np.sqrt(0.2)*R.std(axis=(1,2)).mean(), np.sqrt(0.2*R.var(0) + (1-0.2)*R.var(0)/len(R)).mean()))
        # leave-one-out correlations
        cors=[]
        for i in range(len(R)):
            Ki=(R.sum(0)-R[i])/(len(R)-1)
            a=R[i].ravel(); b=Ki.ravel()
            a=a-a.mean(); b=b-b.mean()
            cors.append(float((a*b).mean()/(a.std()*b.std()+1e-9)))
        print('  LOO corr mean %.4f min %.4f max %.4f'%(np.mean(cors),np.min(cors),np.max(cors)))
        # per-pixel correlation map: corr across images at each pixel
        Rm=R-R.mean(0,keepdims=True)
        sd=Rm.std(0)+1e-6
        P=(Rm/sd)
        # correlation of pixel time series with mean time series? compute consistency: var of mean / mean var
        # squared coherence per pixel: (mean^2 / var) * N approximately
        coh = (K**2) / (R.var(0)+1e-9) * len(R)
        print('  coherence per pixel: mean %.3f median %.3f p99 %.3f max %.3f'%(coh.mean(),np.median(coh),np.percentile(coh,99),coh.max()))
        # sum of pairwise correlations = trace of correlation matrix
        # estimate average pairwise corr via random subset
        idx=np.random.RandomState(0).choice(len(R), size=(300,2))
        pc=[np.corrcoef(R[i].ravel(),R[j].ravel())[0,1] for i,j in idx if i!=j]
        print('  random pairwise corr mean %.4f median %.4f max %.4f'%(np.mean(pc),np.median(pc),np.max(pc)))
        # correlation map at coarse blocks
        H,W=R.shape[1:]
        bs=40
        coh_b=coh.reshape(H//bs,bs,W//bs,bs).mean(axis=(1,3))
        print('  block coherence min %.2f max %.2f'%(coh_b.min(),coh_b.max()))
        # save coherence map
        np.save(f'/home/kozak/mllm/MML/work/solutions/task_japan/coh_{h}.npy', coh_b)

if __name__=='__main__':
    main()
