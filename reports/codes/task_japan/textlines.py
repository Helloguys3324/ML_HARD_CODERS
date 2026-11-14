import numpy as np, glob, os, json
from PIL import Image
from multiprocessing import Pool

ROOT='/home/kozak/mllm/MML/work/tasks/task_japan/data/dataset'

def textlines(p):
    a=np.asarray(Image.open(p).convert('RGB'),dtype=np.int16)
    mx=a.max(2); mn=a.min(2)
    mask=(mx-mn<=3)&(mx>150)
    # remove isolated specks? use row profile
    prof=mask.sum(1)
    # text rows: count > 40 (line of text has many pixels)
    thr=40
    idx=np.where(prof>thr)[0]
    groups=[]
    if len(idx):
        st=idx[0]; prev=idx[0]
        for v in idx[1:]:
            if v-prev>3:
                groups.append((st,prev)); st=v
            prev=v
        groups.append((st,prev))
    groups=[g for g in groups if g[1]-g[0]>=8]
    lines=[]
    for (y0,y1) in groups:
        sub=mask[y0:y1+1]
        xs=np.nonzero(sub.any(0))[0]
        if len(xs)==0: continue
        # require dense-ish line
        if sub.sum()<80: continue
        lines.append(dict(top=int(y0),bot=int(y1),h=int(y1-y0+1),left=int(xs.min()),right=int(xs.max()),npix=int(sub.sum())))
    return dict(rel=os.path.relpath(p,ROOT), lines=lines)

if __name__=='__main__':
    ps=sorted(glob.glob(os.path.join(ROOT,'**','*.jpg'), recursive=True))
    with Pool(4) as pool:
        res=pool.map(textlines, ps, chunksize=4)
    data={r['rel']:r for r in res}
    json.dump(data, open('/home/kozak/mllm/MML/work/solutions/task_japan/textlines.json','w'))
    # print some
    for rel in list(data)[:10]:
        print(rel, [(l['top'],l['h'],l['left'],l['right']) for l in data[rel]['lines']])
