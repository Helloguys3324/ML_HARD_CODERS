import numpy as np, glob, os, collections
from PIL import Image

ROOT='/home/kozak/mllm/MML/work/tasks/task_japan/data/dataset'

def bar_start(p):
    a=np.asarray(Image.open(p).convert('L'),dtype=np.float32)
    rows=(a<8).mean(axis=1)
    # find the first row from bottom region where a long run of near-black rows starts
    dark = rows > 0.85
    # find start of final dark run
    idx = None
    for y in range(len(dark)-1, -1, -1):
        if dark[y]:
            idx = y
        elif idx is not None and not dark[y]:
            break
    return idx if idx is not None else None

def main():
    rows=[]
    for h in sorted(os.listdir(os.path.join(ROOT,'credited'))):
        ps=sorted(glob.glob(os.path.join(ROOT,'credited',h,'*.jpg')))
        vals=[bar_start(p) for p in ps]
        print('%12s'%h, 'bar starts:', vals)
        rows.append((h, vals))
    print()
    for p in sorted(glob.glob(os.path.join(ROOT,'anonymous','*.jpg'))):
        b=bar_start(p)
        rows.append((os.path.basename(p), b))
    # summarize anonymous bar starts histogram
    hs=[b for _,b in rows if isinstance(_,str) and _.startswith('w')]
    print('anon bar start histogram:')
    c=collections.Counter(hs)
    for k in sorted(c): print(' ', k, c[k])

if __name__=='__main__':
    main()
