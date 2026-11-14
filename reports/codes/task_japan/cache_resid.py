import numpy as np, glob, os, sys
from PIL import Image
from scipy import ndimage
from multiprocessing import Pool

ROOT='/home/kozak/mllm/MML/work/tasks/task_japan/data/dataset'
CACHE='/home/kozak/mllm/MML/work/solutions/task_japan/cache'
os.makedirs(CACHE, exist_ok=True)

def one(p):
    name=os.path.relpath(p, ROOT).replace('/','__')
    out=os.path.join(CACHE, name + '.npy')
    if os.path.exists(out): return name
    g=np.asarray(Image.open(p).convert('RGB'), dtype=np.float32).mean(axis=2)
    r=(g-ndimage.gaussian_filter(g,1.2))[20:700,20:1260]
    np.save(out, r.astype(np.float16))
    return name

if __name__=='__main__':
    ps=sorted(glob.glob(os.path.join(ROOT,'**','*.jpg'), recursive=True))
    print(len(ps),'images')
    with Pool(4) as pool:
        for i,_ in enumerate(pool.imap_unordered(one, ps, chunksize=4)):
            if i%25==0: print(i, flush=True)
    print('done')
