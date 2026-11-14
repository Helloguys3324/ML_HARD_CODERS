import numpy as np, csv
from scipy.io import wavfile
import torch
from challenge_model import load_model, predict_probabilities

m = load_model()
torch.set_num_threads(8)
rows = list(csv.DictReader(open('data/recordings.csv')))
label_map = {'birds':0,'frogs':1,'insects':2,'rain':3}
clips=[]
for r in rows:
    rate,smp=wavfile.read('data/'+r['path']); clips.append(smp.astype(np.float32)/32768.0)
audio=np.stack(clips)
true=np.array([label_map[r['label']] for r in rows])
N=32000; sr=16000

def band_noise(lo,hi,seed=7):
    rng=np.random.default_rng(seed)
    n=rng.standard_normal((1,N)).astype(np.float32)
    X=np.fft.rfft(n,axis=-1)
    f=np.fft.rfftfreq(N,1/sr)
    mask=((f>=lo)&(f<hi)).astype(np.float32)
    X=X*mask
    out=np.fft.irfft(X,n=N,axis=-1).astype(np.float32)
    # broadcast
    return np.repeat(out, len(audio), axis=0)

def apply(pert, frac=0.01):
    pert=pert.astype(np.float32)
    ep=(pert**2).sum(-1,keepdims=True); ep[ep==0]=1
    p=pert/np.sqrt(ep)
    sc=np.sqrt(frac*np.maximum((audio**2).sum(-1,keepdims=True),1e-12))
    return audio+p*sc

def report(name, mod):
    e=((mod-audio)**2).sum(-1)/(audio**2).sum(-1)
    pr=predict_probabilities(m,mod).argmax(-1)
    nonrain=true!=3
    fl=((true!=3)&(pr==3)).sum()
    rok=((true==3)&(pr==3)).sum()/max(1,(true==3).sum())
    per=[]
    for c in range(4):
        mask=true==c
        per.append(f"{ (pr[mask]==3).sum()}/{mask.sum()}")
    print(f"[{name}] flip {fl}/{nonrain.sum()} rain_keep {rok:.2f} per(b,f,i,r)={per}")

for lo,hi in [(1000,3000),(3000,6000),(6000,8000),(1000,8000),(2000,5000),(100,8000)]:
    report(f"band {lo}-{hi}Hz", apply(band_noise(lo,hi)))

# comb of rain-like: broad hiss + low rumble
report("hiss 1-8k + rumble <200", apply(band_noise(1000,8000,7)+band_noise(5,200,8)))