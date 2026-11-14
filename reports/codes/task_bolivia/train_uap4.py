import numpy as np, csv, time
from scipy.io import wavfile
import torch
from challenge_model import load_model

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
c = 0.0999

x = torch.from_numpy(audio.copy())
xn = x.norm(dim=-1, keepdim=True)

delta = torch.zeros(N, requires_grad=True, dtype=torch.float32)
torch.manual_seed(123); delta.data = torch.randn(N)*1e-3

def get_pred():
    with torch.no_grad():
        xp = x + c*xn*(delta.reshape(1,N)/(delta.norm()+1e-12))
        return m(xp).argmax(-1).numpy()

def report(it, t0, loss):
    pred = get_pred()
    fl=[]
    for cc in range(4):
        mm=(true==cc); fl.append(f"{pred[mm].tolist().count(3)}/{mm.sum()}")
    fltot=(pred[true!=3]==3).sum()
    rok=(pred[true==3]==3).sum()
    print(f"it {it:3d} t={time.time()-t0:6.1f}s loss={loss.item():.4f} flip {fltot}/23 rain {rok}/8 per(b,f,i,r)={fl}", flush=True)
    return fltot,rok

# Adam with lr schedule
opt = torch.optim.Adam([delta], lr=0.4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=250)

t0=time.time()
for it in range(250):
    opt.zero_grad()
    d = delta.reshape(1,N)
    dn = d/(d.norm()+1e-12)
    xp = x + c*xn*dn
    out = m(xp)
    lp = out.log_softmax(-1)[:,TARGET]
    p = out.softmax(-1)
    w = (1 - p[:,TARGET].detach()).clamp(min=0.02)  # hard-sample weighting
    loss = -(lp*w).sum() / w.sum()
    loss.backward()
    opt.step()
    sched.step()
    if it % 10 == 0 or it==249:
        fltot,rok = report(it, t0, loss)
        if fltot==23 and rok==8:
            dl = delta.detach().numpy().astype(np.float32)
            np.save('uap_rain.npy', dl)
            print("ALL FLIPPED, saved, stop at", it)
            break

dl = delta.detach().numpy().astype(np.float32)
np.save('uap_rain.npy', dl)
print("saved uap_rain.npy norm", np.linalg.norm(dl))
report(999, t0, loss)