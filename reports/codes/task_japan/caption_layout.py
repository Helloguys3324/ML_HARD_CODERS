import numpy as np, glob, os, json, collections
from multiprocessing import Pool

G='/home/kozak/mllm/MML/work/solutions/task_japan/gray'
ROOT='/home/kozak/mllm/MML/work/tasks/task_japan/data/dataset'

def caption_feats(p):
    n=os.path.relpath(p,ROOT).replace('/','__')
    a=np.load(os.path.join(G,n+'.npy')).astype(np.int16)
    H,W=a.shape
    black=(a<=6)
    rowf=black.mean(1)
    # find full-width black bands (rows > 0.9 black)
    band = rowf>0.9
    # find runs of band rows
    runs=[]; start=None
    for i,v in enumerate(band):
        if v and start is None: start=i
        elif not v and start is not None:
            runs.append((start,i)); start=None
    if start is not None: runs.append((start,H))
    # the caption band is the longest run of black rows (if >=60 px)
    runs=[r for r in runs if r[1]-r[0]>=60]
    out=dict(file=os.path.basename(p), pos=None, band_h=0, text_top=None,text_bottom=None,
             text_left=None,text_right=None, lines=0, line_heights=[], align=None, n_white=0)
    if not runs:
        # maybe floating box: look for a black rectangle not full width
        return out
    y0,y1=max(runs,key=lambda r:r[1]-r[0])
    out['band_h']=int(y1-y0)
    out['pos']='top' if y0 < H/2 else 'bottom'
    sub=a[y0:y1]
    white=(sub>110)
    ys,xs=np.nonzero(white)
    if len(xs)<30:
        return out
    out['n_white']=int(len(xs))
    out['text_top']=int(y0+ys.min()); out['text_bottom']=int(y0+ys.max())
    out['text_left']=int(xs.min()); out['text_right']=int(xs.max())
    # line segmentation in the band
    prof=white.sum(1)
    thr=max(4, prof.max()*0.03)
    idx=np.where(prof>thr)[0]
    groups=[]
    if len(idx):
        st=idx[0]; prev=idx[0]
        for v in idx[1:]:
            if v-prev>2:
                groups.append((st,prev)); st=v
            prev=v
        groups.append((st,prev))
    # filter tiny groups
    groups=[g for g in groups if g[1]-g[0]>=6]
    out['lines']=len(groups)
    out['line_heights']=[int(g[1]-g[0]+1) for g in groups]
    # per-line horizontal bbox and alignment
    bbs=[]
    for (ga,gb) in groups:
        sub2=white[ga:gb+1]
        xs2=np.nonzero(sub2.any(0))[0]
        if len(xs2): bbs.append([int(xs2.min()),int(xs2.max())])
    out['line_bbs']=bbs
    if bbs:
        lefts=[b[0] for b in bbs]; rights=[b[1] for b in bbs]
        # alignment by variance of left edges vs right edges
        import statistics
        vl=np.std(lefts); vr=np.std(rights)
        out['align']='left' if vl<vr else 'right'
        out['left_std']=float(vl); out['right_std']=float(vr)
        out['margin_left']=int(min(lefts)); out['margin_right']=int(W-1-max(rights))
    return out

if __name__=='__main__':
    ps=sorted(glob.glob(os.path.join(ROOT,'**','*.jpg'), recursive=True))
    with Pool(4) as pool:
        res=pool.map(caption_feats, ps, chunksize=4)
    out={}
    for p,r in zip(ps,res):
        r['rel']=os.path.relpath(p,ROOT)
        out[r['rel']]=r
    json.dump(out, open('/home/kozak/mllm/MML/work/solutions/task_japan/caption.json','w'), indent=0)
    for h in sorted(os.listdir(os.path.join(ROOT,'credited'))):
        rows=[r for r in out.values() if r['rel'].startswith('credited/'+h+'/')]
        print('='*80); print(h)
        for r in rows:
            print(' %-12s pos=%-6s band=%3d lines=%d heights=%s align=%s margins L%d R%d'%(
                r['file'],str(r['pos']),r['band_h'],r['lines'],r['line_heights'],str(r['align']),
                r.get('margin_left',-1),r.get('margin_right',-1)))
