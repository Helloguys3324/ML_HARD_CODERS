import numpy as np, glob, os, json
from PIL import Image
from scipy import ndimage
from multiprocessing import Pool

ROOT='/home/kozak/mllm/MML/work/tasks/task_japan/data/dataset'

def curve(p):
    a=np.asarray(Image.open(p).convert('RGB'),dtype=np.float32)
    # content region above caption: rows 20..640, cols 20..1260
    x=a[20:640,20:1260]
    g=x.mean(2)
    mu=ndimage.gaussian_filter(g,2)
    # local std of residual at sigma 1
    r=g-ndimage.gaussian_filter(g,1.0)
    sd=np.sqrt(ndimage.gaussian_filter(r*r,2))
    bins=np.arange(0,256,16)
    idx=np.digitize(mu.ravel(),bins)-1
    sdv=sd.ravel()
    med=[]
    for b in range(len(bins)-1):
        m=idx==b
        med.append(float(np.median(sdv[m])) if m.sum()>100 else np.nan)
    # channel noise ratios
    rc=[x[...,i]-ndimage.gaussian_filter(x[...,i],1.0) for i in range(3)]
    ratios=[float(rc[0].std()/(rc[1].std()+1e-9)), float(rc[2].std()/(rc[1].std()+1e-9))]
    # noise correlation between channels
    c01=float(np.corrcoef(rc[0].ravel()[::7],rc[1].ravel()[::7])[0,1])
    c02=float(np.corrcoef(rc[0].ravel()[::7],rc[2].ravel()[::7])[0,1])
    return dict(rel=os.path.relpath(p,ROOT), curve=med, ratios=ratios, corr=[c01,c02], mean=float(g.mean()))

if __name__=='__main__':
    ps=sorted(glob.glob(os.path.join(ROOT,'**','*.jpg'), recursive=True))
    with Pool(4) as pool:
        res=pool.map(curve, ps, chunksize=3)
    json.dump(res, open('/home/kozak/mllm/MML/work/solutions/task_japan/noise_curve.json','w'))
    for h in sorted(os.listdir(os.path.join(ROOT,'credited'))):
        rows=[r for r in res if r['rel'].startswith('credited/'+h+'/')]
        C=np.array([r['curve'] for r in rows],dtype=float)
        ratios=np.array([r['ratios'] for r in rows])
        corr=np.array([r['corr'] for r in rows])
        print('%-12s'%h)
        print('   curve  ', ' '.join('%5.1f'%(np.nanmedian(C[:,i])) for i in range(C.shape[1])))
        print('   ratio  %s  corr %s'%(np.round(ratios.mean(0),4), np.round(corr.mean(0),3)))
    an=[r for r in res if r['rel'].startswith('anonymous')]
    C=np.array([r['curve'] for r in an],dtype=float)
    print('%-12s'%'anon')
    print('   curve  ', ' '.join('%5.1f'%(np.nanmedian(C[:,i])) for i in range(C.shape[1])))
    ratios=np.array([r['ratios'] for r in an]); corr=np.array([r['corr'] for r in an])
    print('   ratio  %s  corr %s'%(np.round(ratios.mean(0),4), np.round(corr.mean(0),3)))
