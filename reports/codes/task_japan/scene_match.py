import numpy as np, glob, os, hashlib, collections
from PIL import Image

G='/home/kozak/mllm/MML/work/solutions/task_japan/gray'
ROOT='/home/kozak/mllm/MML/work/tasks/task_japan/data/dataset'

def names():
    cred={}
    for h in sorted(os.listdir(os.path.join(ROOT,'credited'))):
        cred[h]=[os.path.relpath(p,ROOT).replace('/','__') for p in sorted(glob.glob(os.path.join(ROOT,'credited',h,'*.jpg')))]
    anon=[os.path.relpath(p,ROOT).replace('/','__') for p in sorted(glob.glob(os.path.join(ROOT,'anonymous','*.jpg')))]
    return cred, anon

def thumb(n, size=(64,48)):
    a=np.load(os.path.join(G,n+'.npy')).astype(np.float32)[20:940,20:1260]
    im=Image.fromarray(a.astype(np.uint8)).resize(size, Image.BILINEAR)
    v=np.asarray(im,dtype=np.float32).ravel()
    v-=v.mean(); v/=v.std()+1e-9
    return v

def main():
    cred, anon = names()
    cn=[]; cl=[]
    for h,ns in cred.items():
        for n in ns: cn.append(n); cl.append(h)
    an=[]
    for n in anon: an.append(n)
    T=np.stack([thumb(n) for n in cn+an])
    names_all=cn+an
    labels=cl+['anonymous']*len(an)
    S=T@T.T/T.shape[1]
    np.save('/home/kozak/mllm/MML/work/solutions/task_japan/scene_sim.npy',S)
    import json
    json.dump({'names':names_all,'labels':labels}, open('/home/kozak/mllm/MML/work/solutions/task_japan/scene_index.json','w'))
    n=len(cn)
    # within credited, across-person max
    print('Top credited-anonymous matches:')
    sub=S[n:, :n]
    idx=np.dstack(np.unravel_index(np.argsort(sub.ravel())[::-1], sub.shape))[0]
    seen=set()
    for i,j in idx[:40]:
        if i in seen: continue
        seen.add(i)
        print('  %-22s -> %-15s %-22s %.3f'%(names_all[n+i].replace('anonymous__',''), labels[j], names_all[j].replace('credited__',''), sub[i,j]))
        if len(seen)>=15: break
    # distribution
    print('credited-anon sim: mean %.3f p99 %.3f max %.3f'%(sub.mean(),np.percentile(sub,99),sub.max()))
    # exact duplicate check
    hashes=collections.defaultdict(list)
    for nm in names_all:
        hashes[hashlib.md5(T[names_all.index(nm)].tobytes()).hexdigest()].append(nm)
    dups={k:v for k,v in hashes.items() if len(v)>1}
    print('exact thumbnail dups:', len(dups))

if __name__=='__main__':
    main()
