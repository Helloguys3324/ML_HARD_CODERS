import numpy as np, csv
from scipy.io import wavfile
import torch
from challenge_model import load_model, predict_probabilities

m = load_model(); torch.set_num_threads(8)
rows = list(csv.DictReader(open('data/recordings.csv')))
label_map = {'birds':0,'frogs':1,'insects':2,'rain':3}
clips=[]
for r in rows:
    rate,smp=wavfile.read('data/'+r['path']); clips.append(smp.astype(np.float32)/32768.0)
audio=np.stack(clips)
true=np.array([label_map[r['label']] for r in rows])
pred=predict_probabilities(m,audio).argmax(-1)
N=32000; sr=16000
f=np.fft.rfftfreq(N,1/sr)

S=np.abs(np.fft.rfft(audio,axis=-1))**2
total=S.sum(-1,keepdims=True)+1e-12
def bandf(lo,hi):
    m=(f>=lo)&(f<hi)
    return (S[:,m].sum(-1)/total.squeeze(-1))

feats={}
feats['low_below100']=bandf(0,100)
feats['mid_100_1k']=bandf(100,1000)
feats['mid_1k_3k']=bandf(1000,3000)
feats['high_3_8k']=bandf(3000,8000)
cen=np.sum(f[None,:]*S,axis=-1)/total.squeeze(-1)
feats['centroid']=cen
zc=(np.diff(np.sign(audio),axis=-1)!=0).sum(-1)/N
feats['zcrate']=zc

print(f"{'idx':>3} {'true':>7} {'pred':>7} | "+ " ".join(f"{k[:10]:>10}" for k in feats))
for i in range(len(audio)):
    vals=" ".join(f"{v[i]:10.4f}" for v in feats.values())
    print(f"{i:3d} {['birds','frogs','insects','rain'][true[i]]:>7} {['birds','frogs','insects','rain'][pred[i]]:>7} | {vals}")

print("\n--- group means (fail vs ref within class, 0=birds 1=frogs 2=insects) ---")
for c in range(3):
    fa=[i for i in range(len(audio)) if true[i]==c and pred[i]==3]
    ok=[i for i in range(len(audio)) if true[i]==c and pred[i]==c]
    print(f"class {['birds','frogs','insects'][c]}: fail={fa} ref={ok}")
    for k in feats:
        print(f"   {k:12s} fail={np.mean([feats[k][i] for i in fa]):.4f} ref={np.mean([feats[k][i] for i in ok]):.4f}")