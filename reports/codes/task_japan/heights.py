import numpy as np, glob, os, collections, json
from PIL import Image

ROOT='/home/kozak/mllm/MML/work/tasks/task_japan/data/dataset'

def content_height(p):
    a=np.asarray(Image.open(p).convert('RGB'),dtype=np.int16)
    black = (a.max(axis=2) <= 2)  # near-black pixel
    rows = black.mean(axis=1)
    # find start of final run of rows with >0.98 black
    y = len(rows)-1
    while y>=0 and rows[y] > 0.98:
        y -= 1
    return y+1  # content rows = 0..y, bar from y+1

def main():
    res={}
    for h in sorted(os.listdir(os.path.join(ROOT,'credited'))):
        ps=sorted(glob.glob(os.path.join(ROOT,'credited',h,'*.jpg')))
        vals=[content_height(p) for p in ps]
        res[h]=vals
        print('%12s'%h, vals)
    print()
    anon=sorted(glob.glob(os.path.join(ROOT,'anonymous','*.jpg')))
    av=[content_height(p) for p in anon]
    print('anon content heights histogram:')
    c=collections.Counter(av)
    for k in sorted(c): print(' ', k, c[k])
    json.dump({'credited':res, 'anon':{os.path.basename(p):v for p,v in zip(anon,av)}},
              open('/home/kozak/mllm/MML/work/solutions/task_japan/content_heights.json','w'), indent=1)
    # per credited person unique heights
    for h,v in res.items():
        print('%12s unique %s'%(h, sorted(set(v))))

if __name__=='__main__':
    main()
