import numpy as np, glob, os, json, collections
from PIL import Image
from multiprocessing import Pool

ROOT='/home/kozak/mllm/MML/work/tasks/task_japan/data/dataset'
G='/home/kozak/mllm/MML/work/solutions/task_japan/gray'

def content_bbox(p):
    a=np.load(os.path.join(G, os.path.relpath(p,ROOT).replace('/','__')+'.npy')).astype(np.int16)
    H,W=a.shape
    black=(a<=2)
    # examine upper region rows 0..750 to avoid caption text
    top=black[:750]
    colfrac=top.mean(0)
    rowfrac=black.mean(1)
    # left content edge
    x0=0
    while x0<W//2 and colfrac[x0]>0.75: x0+=1
    x1=W-1
    while x1>W//2 and colfrac[x1]>0.75: x1-=1
    # find bottom of photo: scan from row 750..959 for a long run of rows with >85% black
    # Actually caption bar starts where rowfrac jumps; find first row y in 600..959 where
    # all rows y..y+30 have rowfrac>0.8 (allowing text)
    ybot=None
    for y in range(600,940):
        if rowfrac[y]>0.70 and rowfrac[y:min(y+8,H)].min()>0.55:
            ybot=y; break
    return dict(x0=int(x0),x1=int(x1),ybot=None if ybot is None else int(ybot),
                w=int(x1-x0+1), h=None if ybot is None else int(ybot),
                aspect=None if ybot is None else float((x1-x0+1)/ybot))

if __name__=='__main__':
    ps=sorted(glob.glob(os.path.join(ROOT,'**','*.jpg'), recursive=True))
    with Pool(4) as pool:
        res=pool.map(content_bbox, ps, chunksize=4)
    out={}
    for p,r in zip(ps,res):
        r['rel']=os.path.relpath(p,ROOT)
        out[r['rel']]=r
    json.dump(out, open('/home/kozak/mllm/MML/work/solutions/task_japan/bbox.json','w'), indent=0)
    for h in sorted(os.listdir(os.path.join(ROOT,'credited'))):
        rows=[r for r in out.values() if r['rel'].startswith('credited/'+h+'/')]
        print('%-12s'%h)
        for r in rows:
            print('   %-28s x0=%4d x1=%4d w=%4d h=%s aspect=%s'%(r['rel'].split('/')[-1],r['x0'],r['x1'],r['w'],r['h'],'%.4f'%r['aspect'] if r['aspect'] else None))
