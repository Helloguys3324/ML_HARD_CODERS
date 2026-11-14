import numpy as np, glob, os
from PIL import Image
from multiprocessing import Pool

ROOT='/home/kozak/mllm/MML/work/tasks/task_japan/data/dataset'
CACHE='/home/kozak/mllm/MML/work/solutions/task_japan/gray'
os.makedirs(CACHE, exist_ok=True)

def one(p):
    name=os.path.relpath(p, ROOT).replace('/','__')
    out=os.path.join(CACHE, name + '.npy')
    if os.path.exists(out): return
    g=np.asarray(Image.open(p).convert('L'), dtype=np.uint8)
    np.save(out, g)

if __name__=='__main__':
    ps=sorted(glob.glob(os.path.join(ROOT,'**','*.jpg'), recursive=True))
    with Pool(4) as pool:
        pool.map(one, ps, chunksize=4)
    print('cached', len(ps))
