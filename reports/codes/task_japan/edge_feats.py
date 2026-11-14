import numpy as np, glob, os, json
from PIL import Image
from scipy import ndimage
from multiprocessing import Pool

ROOT='/home/kozak/mllm/MML/work/tasks/task_japan/data/dataset'

def feat(p):
    a=np.asarray(Image.open(p).convert('L'),dtype=np.float32)
    # use content area above y=650 to avoid caption
    g=a[:650, :]
    out={}
    # blockiness: difference across 8x8 boundaries vs interior
    d_h=np.abs(np.diff(g,axis=1))
    cols=np.arange(d_h.shape[1])
    at_boundary=(cols%8==7)
    out['block_h']=float(d_h[:,at_boundary].mean()/ (d_h[:,~at_boundary].mean()+1e-9))
    d_v=np.abs(np.diff(g,axis=0))
    rows=np.arange(d_v.shape[0])
    out['block_v']=float(d_v[rows%8==7,:].mean()/ (d_v[rows%8!=7,:].mean()+1e-9))
    # edge overshoot: gradients
    gy,gx=np.gradient(g)
    mag=np.hypot(gx,gy)
    # strong edges
    thr=np.percentile(mag,99)
    ys,xs=np.nonzero(mag>thr)
    # sample profile perpendicular: use gradient direction approx
    profiles=[]
    rng=np.random.RandomState(0)
    idx=rng.choice(len(ys), size=min(400,len(ys)), replace=False)
    for i in idx:
        y,x=ys[i],xs[i]
        if y<4 or y>=g.shape[0]-4 or x<4 or x>=g.shape[1]-4: continue
        # use horizontal or vertical direction
        if abs(gx[y,x])>abs(gy[y,x]):
            prof=g[y, x-3:x+4]
        else:
            prof=g[y-3:y+4, x]
        if prof.max()-prof.min()<10: continue
        prof=(prof-prof.min())/(prof.max()-prof.min()+1e-9)
        # orient so increasing
        if prof[0]>prof[-1]: prof=prof[::-1]
        profiles.append(prof)
    if profiles:
        P=np.mean(profiles,axis=0)
        out['edge_profile']=P.tolist()
        # overshoot: max beyond 1 at the high end and min below 0 at low end
        out['overshoot_hi']=float(P[5:].max()-1.0)
        out['overshoot_lo']=float(-P[:2].min())
    # noise in flat regions vs textured
    r=g-ndimage.gaussian_filter(g,1.0)
    grad=np.hypot(*np.gradient(g))
    flat=grad<np.percentile(grad,20)
    tex=grad>np.percentile(grad,80)
    out['noise_flat']=float(r[flat].std())
    out['noise_tex']=float(r[tex].std())
    out['noise_ratio']=float(out['noise_flat']/(out['noise_tex']+1e-9))
    # local contrast
    out['rms_contrast']=float(ndimage.gaussian_filter(g,3).std())
    return out

if __name__=='__main__':
    ps=sorted(glob.glob(os.path.join(ROOT,'**','*.jpg'), recursive=True))
    with Pool(4) as pool:
        res=pool.map(feat, ps, chunksize=3)
    data={}
    for p,r in zip(ps,res):
        r['rel']=os.path.relpath(p,ROOT); data[r['rel']]=r
    json.dump(data, open('/home/kozak/mllm/MML/work/solutions/task_japan/edge.json','w'))
    keys=['block_h','block_v','overshoot_hi','overshoot_lo','noise_flat','noise_tex','noise_ratio','rms_contrast']
    print('%-12s'%'person', ' '.join('%10s'%k for k in keys))
    groups=[]
    for h in sorted(os.listdir(os.path.join(ROOT,'credited'))):
        rows=[r for r in data.values() if r['rel'].startswith('credited/'+h+'/')]
        vals=[np.mean([r[k] for r in rows]) for k in keys]
        sds=[np.std([r[k] for r in rows]) for k in keys]
        print('%-12s'%h, ' '.join('%10.3f'%v for v in vals))
        print('%-12s'%'', ' '.join('%10.3f'%v for v in sds))
