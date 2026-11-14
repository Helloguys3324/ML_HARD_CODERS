import numpy as np, glob, os, json
import cv2
from multiprocessing import Pool

G='/home/kozak/mllm/MML/work/solutions/task_japan/gray'
ROOT='/home/kozak/mllm/MML/work/tasks/task_japan/data/dataset'

def analyze(p):
    n=os.path.relpath(p,ROOT).replace('/','__')
    a=np.load(os.path.join(G,n+'.npy')).astype(np.uint8)
    H,W=a.shape
    black=(a<=8).astype(np.uint8)
    # close text holes
    k=cv2.getStructuringElement(cv2.MORPH_RECT,(25,15))
    closed=cv2.morphologyEx(black, cv2.MORPH_CLOSE, k)
    ncomp,lab,stats,cent=cv2.connectedComponentsWithStats(closed,8)
    # find largest component(s) that look like text boxes: area>20000, width>200, height>40
    cands=[]
    for i in range(1,ncomp):
        x,y,w,h,area=stats[i]
        if area>15000 and w>150 and h>35:
            cands.append((area,x,y,w,h,i))
    out=dict(file=os.path.basename(p), boxes=[])
    if not cands:
        return out
    cands.sort(reverse=True)
    for area,x,y,w,h,i in cands[:2]:
        mask=(lab==i)
        # white text inside the (closed) box region
        sub=a[y:y+h, x:x+w]
        submask=mask[y:y+h, x:x+w]
        white=(sub>110)&submask
        ys,xs=np.nonzero(white)
        if len(xs)<30: continue
        # line segmentation
        prof=white.sum(1)
        thr=max(4,prof.max()*0.04)
        idx=np.where(prof>thr)[0]
        groups=[]
        if len(idx):
            st=idx[0]; prev=idx[0]
            for v in idx[1:]:
                if v-prev>2:
                    groups.append((st,prev)); st=v
                prev=v
            groups.append((st,prev))
        groups=[g for g in groups if g[1]-g[0]>=5]
        bbs=[]
        for ga,gb in groups:
            sub2=white[ga:gb+1]
            xs2=np.nonzero(sub2.any(0))[0]
            if len(xs2): bbs.append([int(xs2.min()),int(xs2.max()),int(gb-ga+1)])
        out['boxes'].append(dict(x=int(x),y=int(y),w=int(w),h=int(h),
            n_white=int(len(xs)), lines=len(groups), line_bbs=bbs,
            text_left=int(xs.min()), text_right=int(x+w-1-(w-1-xs.max())) if False else int(xs.max()),
            bbox_w=int(xs.max()-xs.min()+1)))
        break
    return out

if __name__=='__main__':
    ps=sorted(glob.glob(os.path.join(ROOT,'**','*.jpg'), recursive=True))
    with Pool(4) as pool:
        res=pool.map(analyze, ps, chunksize=4)
    out={}
    for p,r in zip(ps,res):
        r['rel']=os.path.relpath(p,ROOT)
        out[r['rel']]=r
    json.dump(out, open('/home/kozak/mllm/MML/work/solutions/task_japan/caption2.json','w'), indent=0)
    for h in sorted(os.listdir(os.path.join(ROOT,'credited'))):
        rows=[r for r in out.values() if r['rel'].startswith('credited/'+h+'/')]
        print('='*70); print(h)
        for r in rows:
            b=r['boxes'][0] if r['boxes'] else None
            if b:
                print(' %-12s box x=%4d y=%4d %4dx%-3d lines=%d bbs=%s'%(r['file'],b['x'],b['y'],b['w'],b['h'],b['lines'],b['line_bbs'][:6]))
            else:
                print(' %-12s NO BOX'%r['file'])
