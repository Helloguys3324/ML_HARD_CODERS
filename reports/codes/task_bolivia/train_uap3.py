import numpy as np, csv, time
from scipy.io import wavfile
import torch
from challenge_model import load_model, predict_probabilities

torch.set_num_threads(8)
m = load_model()
torch.manual_seed(0)
rows = list(csv.DictReader(open('data/recordings.csv')))
label_map = {'birds':0,'frogs':1,'insects':2,'rain':3}
clips=[]
for r in rows:
    rate,smp=wavfile.read('data/'+r['path']); clips.append(smp.astype(np.float32)/32768.0)
audio=np.stack(clips)
true=np.array([label_map[r['label']] for r in rows])
N=32000; TARGET=3
c = 0.0999   # sqrt(0.01)-ish

x = torch.from_numpy(audio.copy())
xn = x.norm(dim=-1, keepdim=True)

delta = torch.zeros(N, requires_grad=True, dtype=torch.float32)
torch.manual_seed(123); delta.data = torch.randn(N)*1e-3

opt = torch.optim.Adam([delta], lr=0.5)

def apply(delta):
    d = delta.reshape(1,N)
    dn = d/(d.norm()+1e-12)
    return x + c*xn*dn

def eval_all():
    with torch.no_grad():
        xp = apply(delta)
        pred = m(xp).argmax(-1).numpy()
    flips=[]
    for cc in range(4):
        mm=(true==cc); flips.append(f"{pred[mm].tolist().count(3)}/{mm.sum()}")
    return pred, flips

best = None
t0=time.time()
for it in range(120):
    opt.zero_grad()
    xp = apply(delta)
    out = m(xp)
    logp = out.log_softmax(-1)
    loss = -logp[:,TARGET].mean()
    loss.backward()
    opt.step()
    if it % 5==0:
        pred, flips = eval_all()
        fl=(pred[true!=3]==3).sum()
        rok=(pred[true==3]==3).sum()
        print(f"it {it:3d} t={time.time()-t0:6.1f}s loss={loss.item():.4f} flip {fl}/23 rain_keep {rok}/8 per(b,f,i,r)={flips}", flush=True)
        if fl==23 and rok==8:
            best = delta.detach().numpy().copy()
            print("ALL FLIPPED, early stop")
            break

dl = delta.detach().numpy().astype(np.float32)
np.save('uap_rain.npy', dl)
print("saved uap_rain.npy norm", np.linalg.norm(dl))

# final report
pred, flips = eval_all()
print("FINAL", flips)