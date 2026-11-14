import numpy as np, glob, os, re
from PIL import Image
from multiprocessing import Pool

ROOT='/home/kozak/mllm/MML/work/tasks/task_japan/data/dataset'

def scan(p):
    a=np.asarray(Image.open(p).convert('RGB'),dtype=np.uint8)
    out={'rel':os.path.relpath(p,ROOT),'hits':[]}
    for ch in range(3):
        bits=(a[...,ch]&1).astype(np.uint8).ravel()
        # pack bits to bytes MSB first
        n=len(bits)//8*8
        by=np.packbits(bits[:n]).tobytes()
        for pat in [b'SIGMOID', b'FLAG', b'flag', b'toorisugari', b'TOORI', b'http', b'.txt', b'PNG', b'JFIF']:
            if pat in by:
                out['hits'].append((ch,pat.decode(),by.find(pat)))
        # printable runs
        txt=re.findall(rb'[ -~]{12,}', by)
        if txt:
            out['runs']=[t[:80].decode('latin1') for t in txt[:5]]
    return out

if __name__=='__main__':
    ps=sorted(glob.glob(os.path.join(ROOT,'**','*.jpg'), recursive=True))
    with Pool(4) as pool:
        res=pool.map(scan, ps, chunksize=4)
    n=0
    for r in res:
        if r['hits'] or r.get('runs'):
            n+=1
            print(r['rel'], r['hits'], r.get('runs'))
    print('files with LSB text-ish content:', n, '/', len(res))
