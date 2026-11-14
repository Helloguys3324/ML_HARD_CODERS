import numpy as np, glob, os, json, collections
from multiprocessing import Pool

G='/home/kozak/mllm/MML/work/solutions/task_japan/gray'
ROOT='/home/kozak/mllm/MML/work/tasks/task_japan/data/dataset'

def content_region(p):
    n=os.path.relpath(p,ROOT).replace('/','__')
    a=np.load(os.path.join(G,n+'.npy')).astype(np.int16)
    H,W=a.shape
    black=(a<=6)
    rowf=black.mean(1); colf=black.mean(0)
    # Find the largest run of rows with black fraction < 0.8 (content rows)
    def runs(frac, thr=0.8, minlen=100):
        out=[]; start=None
        for i,v in enumerate(frac):
            if v<thr:
                if start is None: start=i
            else:
                if start is not None and i-start>=minlen: out.append((start,i))
                start=None
        if start is not None and len(frac)-start>=minlen: out.append((start,len(frac)))
        return out
    rr=runs(rowf); cc=runs(colf)
    if not rr or not cc:
        return dict(y0=0,y1=H,x0=0,x1=W,w=W,h=H,aspect=W/H)
    y0,y1=max(rr,key=lambda t:t[1]-t[0])
    x0,x1=max(cc,key=lambda t:t[1]-t[0])
    w=x1-x0; h=y1-y0
    return dict(y0=int(y0),y1=int(y1),x0=int(x0),x1=int(x1),w=int(w),h=int(h),aspect=float(w/h))

if __name__=='__main__':
    ps=sorted(glob.glob(os.path.join(ROOT,'**','*.jpg'), recursive=True))
    with Pool(4) as pool:
        res=pool.map(content_region, ps, chunksize=4)
    out={}
    for p,r in zip(ps,res):
        r['rel']=os.path.relpath(p,ROOT)
        out[r['rel']]=r
    json.dump(out, open('/home/kozak/mllm/MML/work/solutions/task_japan/content_region.json','w'), indent=0)
    for h in sorted(os.listdir(os.path.join(ROOT,'credited'))):
        rows=[r for r in out.values() if r['rel'].startswith('credited/'+h+'/')]
        print('%-12s'%h)
        for r in rows:
            print('   %-12s  y %4d-%4d  x %4d-%4d  %4dx%-4d  aspect %.4f'%(r['rel'].split('/')[-1],r['y0'],r['y1'],r['x0'],r['x1'],r['w'],r['h'],r['aspect']))
    print()
    print('ANON:')
    for r in out.values():
        if r['rel'].startswith('anonymous'):
            print('   %-12s  y %4d-%4d  x %4d-%4d  %4dx%-4d  aspect %.4f'%(r['rel'].split('/')[-1],r['y0'],r['y1'],r['x0'],r['x1'],r['w'],r['h'],r['aspect']))
