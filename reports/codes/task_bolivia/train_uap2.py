import numpy as np, csv, time
from scipy.io import wavfile
import torch
from challenge_model import load_model, predict_probabilities

m = load_model()
torch.set_num_threads(8)
torch.manual_seed(0)
rows = list(csv.DictReader(open('data/recordings.csv')))
label_map = {'birds':0,'frogs':1,'insects':2,'rain':3}
clips=[]
for r in rows:
    rate,smp=wavfile.read('data/'+r['path']); clips.append(smp.astype(np.float32)/32768.0)
audio=np.stack(clips)
true=np.array([label_map[r['label']] for r in rows])
N=32000
TARGET=3
alpha=0.0999   # sqrt(0.01)

x = torch.from_numpy(audio.copy())
xn = x.norm(dim=-1, keepdim=True)

scale_by_energy = True
if scale_by_energy:
    # unit direction scaled per-sample: alpha * xn
    pass

delta = torch.zeros(N, requires_grad=True, dtype=torch.float32)
torch.manual_seed(123)
delta.data = torch.randn(N) * 1e-4

opt = torch.optim.Adam([delta], lr=0.5)

def evaluate():
    with torch.no_grad():
        d = delta.reshape(1,N)
        dn = d/(d.norm()+1e-12)
        sc = alpha * xn * dn
        xp = x + sc
        pr = m(xp).softmax(-1)
        pred = pr.argmax(-1).numpy()
        inv = pred==TARGET
        return pred, inv

def flips(pred):
    nonrain = true!=3
    details=[]
    for c in range(4):
        mask=true==c
        ok=(pred[mask]==3).sum()
        details.append(f"{ok}/{mask.sum()}")
    return details

best=None
t0=time.time()
for it in range(300):
    opt.zero_grad()
    d = delta.reshape(1,N)
    dn = d/(d.norm()+1e-12)
    sc = alpha * xn * dn
    xp = x + sc
    out = m(xp)
    logp = out.log_softmax(-1)
    loss = -logp[:,TARGET].mean()
    loss.backward()
    opt.step()
    if it % 10 == 0 or it==299:
        pred, inv = evaluate()
        nonrain=(true!=3)
        fl = inv[nonrain].sum()
        print(f"it {it:3d} t={time.time()-t0:5.1f}s loss={loss.item():.4f} "
              f"flip {fl}/{nonrain.sum()} per={flips(pred)}", flush=True)

d = delta.detach().numpy().astype(np.float32)
np.save('uap_rain.npy', d)
print("saved uap_rain.npy", np.linalg.norm(d))