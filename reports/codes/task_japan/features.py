import numpy as np, glob, os, json
from PIL import Image
from scipy import ndimage
from multiprocessing import Pool

ROOT='/home/kozak/mllm/MML/work/tasks/task_japan/data/dataset'
OUT='/home/kozak/mllm/MML/work/solutions/task_japan/features.json'

def feats(p):
    im=Image.open(p).convert('RGB')
    a=np.asarray(im, dtype=np.float32)
    H,W,_=a.shape
    # content region: top part only, avoid caption bar (bar top unknown; use y<700 conservative)
    c=a[20:700,20:1260]
    flat=c.reshape(-1,3)
    out={}
    out['file']=os.path.basename(p)
    out['rel']=os.path.relpath(p,ROOT)
    out['size']=os.path.getsize(p)
    # color stats
    out['mean']=flat.mean(0).tolist()
    out['std']=flat.std(0).tolist()
    qs=[1,5,25,50,75,95,99]
    q=np.percentile(flat,qs,axis=0)
    out['q']=q.tolist()
    mx=c.max(2); mn=c.min(2)
    sat=(mx-mn)/(mx+1e-6)
    out['sat_mean']=float(sat.mean()); out['sat_std']=float(sat.std())
    out['sat_q']=np.percentile(sat,[10,50,90]).tolist()
    # hue-ish ratios
    out['rg']=float((flat[:,0]-flat[:,1]).mean())
    out['gb']=float((flat[:,1]-flat[:,2]).mean())
    out['rb']=float((flat[:,0]-flat[:,2]).mean())
    # luminance
    y=a.mean(2)
    out['y_mean']=float(y.mean())
    # noise: residual std at multiple scales in luminance, content region
    g=y[20:700,20:1260]
    for s in [0.6,1.2,2.5]:
        r=g-ndimage.gaussian_filter(g,s)
        out['noise_%.1f'%s]=float(r.std())
    # noise per channel (chroma noise) at sigma 1.0
    for i,ch in enumerate('RGB'):
        r=c[...,i]-ndimage.gaussian_filter(c[...,i],1.0)
        out['noise_'+ch]=float(r.std())
    # sharpness: gradient energy
    gy,gx=np.gradient(g)
    out['grad_mean']=float(np.sqrt(gx*gx+gy*gy).mean())
    out['grad_p99']=float(np.percentile(np.sqrt(gx*gx+gy*gy),99))
    # laplacian variance
    out['lap_var']=float(ndimage.laplace(g).var())
    # high frequency energy ratio
    F=np.abs(np.fft.rfft2(g-g.mean()))**2
    out['hf_ratio']=float(F[:, F.shape[1]//4:].sum()/ (F.sum()+1e-9))
    # caption bar top: robust detection using exact black background with text
    rgb=a.astype(np.int16)
    black=(rgb.max(2)<=2)
    rows=black.mean(1)
    # find first row of the final long run allowing text rows (text rows have black fraction >0.4 too)
    bar = rows > 0.35
    y0=len(bar)-1
    while y0>=0 and bar[y0]: y0-=1
    bar_top = y0+1
    out['bar_top']=int(bar_top)
    # text mask in bottom bar: near-white pixels
    bottom=a[bar_top:,:,:]
    white = (bottom.min(2)>170)
    ys,xs=np.nonzero(white)
    if len(xs)>50 and len(ys)>20:
        out['text_top']=int(bar_top+ys.min()); out['text_bottom']=int(bar_top+ys.max())
        out['text_left']=int(xs.min()); out['text_right']=int(xs.max())
        # line count: rows with many white pixels
        lineprof=white.sum(1)
        lines=0; inl=False
        for v in lineprof:
            if v>8 and not inl: lines+=1; inl=True
            elif v<=2: inl=False
        out['lines']=lines
        # alignment: compare left margin and right margin
        out['margin_left']=int(xs.min()); out['margin_right']=int(W-1-xs.max())
        # per-line bbox
        lp=white.sum(1)
        idx=np.where(lp>8)[0]
        if len(idx):
            groups=[]; start=idx[0]; prev=idx[0]
            for v in idx[1:]:
                if v-prev>3: groups.append((start,prev)); start=v
                prev=v
            groups.append((start,prev))
            bbs=[]
            for (y1,y2) in groups:
                sub=white[y1:y2+1]
                xs2=np.nonzero(sub.any(0))[0]
                if len(xs2): bbs.append([int(xs2.min()),int(xs2.max()),int(y2-y1+1)])
            out['line_bboxes']=bbs
    else:
        out['text_top']=None; out['lines']=0
    return out

if __name__=='__main__':
    ps=sorted(glob.glob(os.path.join(ROOT,'**','*.jpg'), recursive=True))
    print(len(ps),'images')
    with Pool(4) as pool:
        res=pool.map(feats, ps, chunksize=3)
    json.dump(res, open(OUT,'w'), indent=0)
    print('wrote',OUT)
