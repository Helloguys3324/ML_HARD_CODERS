import numpy as np, glob, os, json
from PIL import Image
from scipy import ndimage

ROOT='/home/kozak/mllm/MML/work/tasks/task_japan/data/dataset'

def stats(p):
    g=np.asarray(Image.open(p).convert('L'),dtype=np.float32)
    c=g[20:700,20:1260]
    r=c-ndimage.gaussian_filter(c,1.2)
    # radial power spectrum of residual
    F=np.abs(np.fft.fftshift(np.fft.fft2(r-r.mean())))**2
    h,w=F.shape; cy,cx=h//2,w//2
    yy,xx=np.ogrid[:h,:w]
    rad=np.sqrt((yy-cy)**2+(xx-cx)**2).astype(int)
    nb=rad.max()+1
    prof=np.bincount(rad.ravel(),F.ravel(),minlength=nb)
    cnt=np.bincount(rad.ravel(),minlength=nb)+1e-9
    prof=prof/cnt
    return dict(std=float(r.std()), nbytes=os.path.getsize(p), spec=prof[:60].tolist(), mean=float(c.mean()), p99=float(np.percentile(c,99)))

def main():
    out={'credited':{}, 'anon':{}}
    for h in sorted(os.listdir(os.path.join(ROOT,'credited'))):
        ps=sorted(glob.glob(os.path.join(ROOT,'credited',h,'*.jpg')))
        out['credited'][h]=[stats(p) for p in ps]
    out['anon']= {os.path.basename(p): stats(p) for p in sorted(glob.glob(os.path.join(ROOT,'anonymous','*.jpg')))}
    json.dump(out, open('/home/kozak/mllm/MML/work/solutions/task_japan/spectra.json','w'))
    print('credited residual std by person:')
    for h,v in out['credited'].items():
        print(' %12s %.2f'%(h, np.mean([s['std'] for s in v])))
    an=np.array([s['std'] for s in out['anon'].values()])
    print('anon residual std mean %.2f median %.2f range %.2f-%.2f'%(an.mean(),np.median(an),an.min(),an.max()))

if __name__=='__main__':
    main()
