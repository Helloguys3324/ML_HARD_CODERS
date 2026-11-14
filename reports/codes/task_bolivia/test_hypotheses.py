import numpy as np, csv
from scipy.io import wavfile
import torch
from challenge_model import load_model, predict_probabilities, CLASS_NAMES

m = load_model()
torch.set_num_threads(8)
rows = list(csv.DictReader(open('data/recordings.csv')))
label_map = {'birds':0,'frogs':1,'insects':2,'rain':3}
clips=[]
for r in rows:
    rate,smp=wavfile.read('data/'+r['path']); clips.append(smp.astype(np.float32)/32768.0)
audio=np.stack(clips)
true=np.array([label_map[r['label']] for r in rows])
energies=(audio**2).sum(-1)
pred0 = predict_probabilities(m,audio).argmax(-1)
N=32000; sr=16000

def lp_noise(corner, seed=0, shape_lowfreq=True):
    # generate noise, filter to low-pass
    rng=np.random.default_rng(seed)
    n=rng.standard_normal(audio.shape).astype(np.float32)
    X=np.fft.rfft(n, axis=-1)
    f=np.fft.rfftfreq(N,1/sr)
    # low-pass mask with 1/f shaping option
    if shape_lowfreq:
        mask=(np.maximum(f, 5.0)**-1.0)  # 1/f above 5Hz
        mask[f<5]=1/5.0
    else:
        mask=np.ones_like(f)
    mask[f>corner]=0.0
    X=X*mask
    out=np.fft.irfft(X,n=N,axis=-1).astype(np.float32)
    return out

def apply(x, pert, frac=0.01):
    pert = pert.astype(np.float32)
    e_p = (pert**2).sum(-1,keepdims=True)
    e_p[e_p==0]=1
    # normalize per row to unit energy then scale sqrt(frac)
    p = pert / np.sqrt(e_p)
    sc = np.sqrt(frac*np.maximum((x**2).sum(-1,keepdims=True),1e-12))
    return x + p*sc

def report(name, mod):
    e=((mod-audio)**2).sum(-1)/energies
    pr=predict_probabilities(m,mod).argmax(-1)
    nonrain=true!=3
    fl=((true!=3)&(pr==3)).sum()
    rok=((true==3)&(pr==3)).sum()
    per={}
    for c in range(3):
        mask=true==c
        per[c]=((pr[mask]==3).sum(), mask.sum())
    print(f"[{name}] maxE={e.max()*100:.3f}% flip nonrain->rain {fl}/{nonrain.sum()} rain_ok {rok}/{(true==3).sum()}  per={[f'{per[c][0]}/{per[c][1]}' for c in range(3)]}")
    return pr

for corner in [20,40,60,100,150,300,1000]:
    for shape in [True]:
        report(f"lp1/f noise {corner}Hz", apply(audio, lp_noise(corner, shape_lowfreq=shape)))
print()
for corner in [40,100,200]:
    report(f"lp flat noise {corner}Hz", apply(audio, lp_noise(corner, shape_lowfreq=False)))

# Also DC-only again
report("DC only", apply(audio, np.ones_like(audio)))