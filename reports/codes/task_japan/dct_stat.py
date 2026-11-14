import numpy as np, glob, os, json
from PIL import Image
from scipy import ndimage
from multiprocessing import Pool

ROOT='/home/kozak/mllm/MML/work/tasks/task_japan/data/dataset'

def dct_feats(p):
    a=np.asarray(Image.open(p).convert('YCbCr'),dtype=np.float32)
    Y=a[...,0]; Cb=a[...,1]-128; Cr=a[...,2]-128
    out={}
    # chroma/luma high-pass energy ratio
    for nm,ch in [('Y',Y),('Cb',Cb),('Cr',Cr)]:
        r=ch-ndimage.gaussian_filter(ch,1.0)
        out['hp_'+nm]=float(r.std())
    out['chroma_ratio']=float((out['hp_Cb']+out['hp_Cr'])/(2*out['hp_Y']+1e-9))
    # DCT coefficient histograms for luma, low-mid frequencies
    y=Y.copy()
    H,W=y.shape
    hh=H//8*8; ww=W//8*8
    blocks=y[:hh,:ww].reshape(hh//8,8,ww//8,8).transpose(0,2,1,3)
    # vectorized DCT via matrix
    dctmtx=np.zeros((8,8),dtype=np.float32)
    for k in range(8):
        for n in range(8):
            dctmtx[k,n]=np.cos(np.pi*(2*n+1)*k/16)*(np.sqrt(1/8) if k==0 else np.sqrt(2/8))
    B=blocks.reshape(-1,8,8)
    C=np.einsum('ij,njk,lk->nil', dctmtx, B, dctmtx)  # n,i,l
    # sample coefficients at frequencies
    freqs=[(0,0),(1,1),(2,1),(1,2),(2,2),(3,1),(3,3),(4,4),(1,0),(0,1)]
    hists={}
    for (i,j) in freqs:
        v=C[:,i,j]
        # histogram of values clipped +-60
        h,_=np.histogram(v,bins=np.arange(-60.5,61.5,1.0))
        hists['dct_%d_%d'%(i,j)]=h
    # periodicity of histogram for each freq: FFT power at periods 2..20
    out['hist']={k: v.tolist() for k,v in hists.items()}
    return out

if __name__=='__main__':
    ps=sorted(glob.glob(os.path.join(ROOT,'**','*.jpg'), recursive=True))
    with Pool(4) as pool:
        res=pool.map(dct_feats, ps, chunksize=2)
    data={}
    for p,r in zip(ps,res):
        r['rel']=os.path.relpath(p,ROOT)
        data[r['rel']]=r
    json.dump(data, open('/home/kozak/mllm/MML/work/solutions/task_japan/dct.json','w'))
    # summarize chroma ratios
    for h in sorted(os.listdir(os.path.join(ROOT,'credited'))):
        rows=[r for r in data.values() if r['rel'].startswith('credited/'+h+'/')]
        print('%-12s chroma_ratio mean %.4f std %.4f  hpY %.2f'%(h,np.mean([r['chroma_ratio'] for r in rows]),np.std([r['chroma_ratio'] for r in rows]),np.mean([r['hp_Y'] for r in rows])))
    an=[r for r in data.values() if r['rel'].startswith('anonymous')]
    print('anon ratio %.4f std %.4f'%(np.mean([r['chroma_ratio'] for r in an]),np.std([r['chroma_ratio'] for r in an])))
