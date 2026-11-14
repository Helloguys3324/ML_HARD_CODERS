import numpy as np, glob, os
from PIL import Image
from scipy import ndimage

ROOT='/home/kozak/mllm/MML/work/tasks/task_japan/data/dataset'
OUT='/home/kozak/mllm/MML/work/solutions/task_japan'

def res(path, s=1.2, crop=(20,700,20,1260)):
    g=np.asarray(Image.open(path).convert('RGB'),dtype=np.float32).mean(axis=2)
    r=g-ndimage.gaussian_filter(g,s)
    y0,y1,x0,x1=crop
    return r[y0:y1,x0:x1]

def ncc(x,y):
    x=x.ravel()-x.mean(); y=y.ravel()-y.mean()
    return float((x*y).mean()/(x.std()*y.std()+1e-12))

handles=sorted(os.listdir(os.path.join(ROOT,'credited')))
# build fingerprints from crop above caption
fps={}
for h in handles:
    ps=sorted(glob.glob(os.path.join(ROOT,'credited',h,'*.jpg')))
    acc=None
    for p in ps:
        r=res(p); acc = r if acc is None else acc+r
    fps[h]=acc/len(ps)

# LOO accuracy per image
correct=0; tot=0; gaps=[]
for h in handles:
    for p in sorted(glob.glob(os.path.join(ROOT,'credited',h,'*.jpg'))):
        r=res(p)
        sc={hh:ncc(r,fps[hh]) for hh in handles}
        best=max(sc,key=sc.get); tot+=1; correct+= (best==h)
        gaps.append(sc[h]-max(v for k,v in sc.items() if k!=h))
print('LOO crop-above-caption: %d/%d  mean gap %.4f'%(correct,tot,np.mean(gaps)))
print('per person own corr with fingerprint:')
for h in handles:
    vals=[ncc(res(p),fps[h]) for p in sorted(glob.glob(os.path.join(ROOT,'credited',h,'*.jpg')))]
    print(' %12s %.4f'%(h,np.mean(vals)))

np.savez(os.path.join(OUT,'fps_content.npz'), **fps)

# anonymous correlation with these fingerprints
anon=sorted(glob.glob(os.path.join(ROOT,'anonymous','*.jpg')))
best_counts={}
sc_all=[]
for p in anon:
    r=res(p)
    sc={h:ncc(r,fps[h]) for h in handles}
    sc_all.append([sc[h] for h in handles])
    b=max(sc,key=sc.get); best_counts[b]=best_counts.get(b,0)+1
print('anon best counts', best_counts)
print('anon score matrix mean/max per person:')
A=np.array(sc_all)
for i,h in enumerate(handles):
    print(' %12s mean %.4f max %.4f'%(h, A[:,i].mean(), A[:,i].max()))
