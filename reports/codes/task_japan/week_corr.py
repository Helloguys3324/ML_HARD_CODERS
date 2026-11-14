import numpy as np, glob, os, collections

CACHE='/home/kozak/mllm/MML/work/solutions/task_japan/cache'
ROOT='/home/kozak/mllm/MML/work/tasks/task_japan/data/dataset'

def load(name):
    return np.load(os.path.join(CACHE, name+'.npy')).astype(np.float32)

def anon_names():
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(ROOT,'anonymous','*.jpg')))

def cred_names():
    out={}
    for d in sorted(os.listdir(os.path.join(ROOT,'credited'))):
        ps=sorted(glob.glob(os.path.join(ROOT,'credited',d,'*.jpg')))
        out[d]=[os.path.basename(p) for p in ps]
    return out

def corr_matrix(names):
    X=[]
    for n in names:
        r=load('anonymous__'+n if os.path.exists(os.path.join(CACHE,'anonymous__'+n+'.npy')) else n)
        X.append(r.ravel())
    X=np.stack(X); X-=X.mean(1,keepdims=True); X/=X.std(1,keepdims=True)+1e-9
    return X@X.T/X.shape[1]

if __name__=='__main__':
    anon=anon_names()
    weeks=collections.defaultdict(list)
    for n in anon: weeks[int(n[1:3])].append(n)
    for wk in sorted(weeks):
        names=weeks[wk]
        M=corr_matrix(names)
        off=M[np.triu_indices(len(M),1)]
        print('week',wk,'n',len(M),'offdiag mean %.4f std %.4f max %.4f'%(off.mean(),off.std(),off.max()))
        idx=np.dstack(np.unravel_index(np.argsort(M.ravel())[::-1],M.shape))[0]
        shown=0
        for i,j in idx:
            if i<j:
                print('   %s %s %.3f'%(names[i],names[j],M[i,j])); shown+=1
                if shown>=5: break
