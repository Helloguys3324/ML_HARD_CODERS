import numpy as np, glob, os, sys
from PIL import Image
from scipy import ndimage

ROOT='/home/kozak/mllm/MML/work/tasks/task_japan/data/dataset'
OUT='/home/kozak/mllm/MML/work/solutions/task_japan'

def res(path, crop=(20,700,20,1260), step=2):
    g=np.asarray(Image.open(path).convert('RGB'),dtype=np.float32).mean(axis=2)
    y0,y1,x0,x1=crop
    g=g[y0:y1:step,x0:x1:step]
    return (g-ndimage.gaussian_filter(g,1.0)).astype(np.float32)

def collect(paths):
    X=[]
    for p in paths:
        r=res(p)
        X.append(r.ravel())
    X=np.stack(X)
    X-=X.mean(axis=1,keepdims=True)
    X/= (X.std(axis=1,keepdims=True)+1e-9)
    return X

def corr_matrix(X):
    n=len(X); M=np.zeros((n,n),dtype=np.float32)
    for i in range(n):
        M[i]=X@X[i]/X.shape[1]
    return M

def main():
    anon=sorted(glob.glob(os.path.join(ROOT,'anonymous','*.jpg')))
    Xa=collect(anon)
    Ma=corr_matrix(Xa)
    np.save(os.path.join(OUT,'anon_corr.npy'),Ma)
    off=Ma[np.triu_indices(len(Ma),1)]
    print('ANON pairwise corr: mean %.5f std %.5f p99 %.4f max %.4f'%(off.mean(),off.std(),np.percentile(off,99),off.max()))
    # top pairs
    idx=np.dstack(np.unravel_index(np.argsort(Ma.ravel())[::-1], Ma.shape))[0]
    seen=set(); print('top pairs:')
    for i,j in idx:
        if i>=j or (i,j) in seen: continue
        seen.add((i,j)); print('  %s %s %.4f'%(os.path.basename(anon[i]),os.path.basename(anon[j]),Ma[i,j]))
        if len(seen)>=10: break
    # PCA
    Xc=Xa-Xa.mean(axis=0,keepdims=True)
    # use SVD on covariance via Gram matrix
    G=Xa@Xa.T/ Xa.shape[1]
    w,V=np.linalg.eigh(G.astype(np.float64))
    w=w[::-1]; V=V[:,::-1]
    print('anon top eigenvalues:', np.round(w[:10],4))
    # credited control
    ps=[]; labels=[]
    for h in sorted(os.listdir(os.path.join(ROOT,'credited'))):
        for p in sorted(glob.glob(os.path.join(ROOT,'credited',h,'*.jpg'))):
            ps.append(p); labels.append(h)
    Xc2=collect(ps)
    Mc=corr_matrix(Xc2)
    np.save(os.path.join(OUT,'cred_corr.npy'),Mc)
    offc=Mc[np.triu_indices(len(Mc),1)]
    print('CRED pairwise corr: mean %.5f std %.5f max %.4f'%(offc.mean(),offc.std(),offc.max()))
    # within vs between block
    lab=np.array(labels)
    same=(lab[:,None]==lab[None,:]); np.fill_diagonal(same,False)
    tri=np.triu_indices(len(lab),1)
    print('CRED within-block mean %.4f, between-block mean %.4f'%(Mc[tri][same[tri]].mean(), Mc[tri][~same[tri]].mean()))
    G2=Xc2@Xc2.T/Xc2.shape[1]
    w2,V2=np.linalg.eigh(G2.astype(np.float64)); w2=w2[::-1]
    print('cred top eigenvalues:', np.round(w2[:10],4))
    np.save(os.path.join(OUT,'cred_labels.npy'),lab)

if __name__=='__main__':
    main()
