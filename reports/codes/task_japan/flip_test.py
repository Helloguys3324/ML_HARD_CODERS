import numpy as np, glob, os
from PIL import Image
from scipy import ndimage

ROOT='/home/kozak/mllm/MML/work/tasks/task_japan/data/dataset'
OUT='/home/kozak/mllm/MML/work/solutions/task_japan'

def res(path, s=1.2, crop=(20,700,20,1260), scale=1):
    g=np.asarray(Image.open(path).convert('RGB'),dtype=np.float32).mean(axis=2)
    y0,y1,x0,x1=crop
    g=g[y0:y1,x0:x1]
    if scale>1:
        g=g[::scale,::scale]
        g=g-ndimage.gaussian_filter(g,1.2)
    else:
        g=g-ndimage.gaussian_filter(g,s)
    return g

def ncc(x,y):
    x=x.ravel()-x.mean(); y=y.ravel()-y.mean()
    return float((x*y).mean()/(x.std()*y.std()+1e-12))

d=np.load(os.path.join(OUT,'fps_content.npz'))
fps={k:d[k] for k in d.files}
handles=sorted(fps)
anon=sorted(glob.glob(os.path.join(ROOT,'anonymous','*.jpg')))

# test transformations on first 8 anonymous images
print('NCC with transformed fingerprints (anon first 8):')
for p in anon[:8]:
    r=res(p)
    best=[]
    for h in handles:
        f=fps[h]
        for name, ff in [('id',f),('flipx',f[:,::-1]),('flipy',f[::-1,:]),('rot180',f[::-1,::-1]),('transpose',f.T)]:
            if ff.shape==r.shape:
                best.append((ncc(r,ff), h, name))
    best.sort(reverse=True)
    print(os.path.basename(p), ['%s/%s %.4f'%(h,n,v) for v,h,n in best[:3]])
