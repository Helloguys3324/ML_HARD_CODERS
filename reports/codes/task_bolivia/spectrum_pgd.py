import numpy as np, csv
from scipy.io import wavfile
import torch
from challenge_model import load_model

m = load_model()
torch.set_num_threads(8)
rows = list(csv.DictReader(open('data/recordings.csv')))
label_map = {'birds':0,'frogs':1,'insects':2,'rain':3}
clips=[]
for r in rows:
    rate,smp=wavfile.read('data/'+r['path']); clips.append(smp.astype(np.float32)/32768.0)
audio=np.stack(clips)
true=np.array([label_map[r['label']] for r in rows])
N=32000

TARGET=3
def pgd_dir(x0, steps=30, lr=0.03):
    x = torch.tensor(x0[None], dtype=torch.float32).clone().requires_grad_(True)
    xb = torch.tensor(x0[None], dtype=torch.float32)
    xn = xb.norm()
    budget = 0.1 * xn
    for i in range(steps):
        out = m(x)
        g = torch.autograd.grad(-out.log_softmax(-1)[:,TARGET].mean(), x)[0]
        x = x - lr * g
        d = x - xb
        dn = d.norm()
        if dn > budget:
            d = d/budget if False else d/dn*budget
        x = (xb + d).detach().clone().requires_grad_(True)
    d = (x - xb).detach().numpy()[0]
    return d

# compute PGd dir for a few samples, collect, analyze spectrum
dirs = []
for idx in [2,6,3,14,10,25]:  # 2 per non-rain class
    dirs.append(pgd_dir(audio[idx]))
dirs = np.stack(dirs)
print("pgd dirs shape", dirs.shape)
# spectrum of each
f = np.fft.rfftfreq(N, 1/16000)
for k,idx in enumerate([2,6,3,14,10,25]):
    X = np.abs(np.fft.rfft(dirs[k]))
    # dominant frequencies (top 10 by magnitude)
    order = np.argsort(X)[::-1]
    tops = [f"{f[j]:.0f}" for j in order[:10]]
    print(f"idx {idx} true={['birds','frogs','insects','rain'][true[idx]]} top freqs {tops}")

# correlation between dirs
d = dirs / (np.linalg.norm(dirs,axis=-1,keepdims=True)+1e-12)
corr = d @ d.T
print("\ncosine sim matrix:\n", np.round(corr,2))

# average direction
avg = dirs.mean(0)
X = np.abs(np.fft.rfft(avg))
order = np.argsort(X)[::-1]
print("\nAVG dir top freqs:", [f"{f[j]:.0f}" for j in order[:20]])
# fraction of energy below 100Hz
elow = (np.abs(np.fft.rfft(avg))**2)[f<100].sum() / (np.abs(np.fft.rfft(avg))**2).sum()
print(f"fraction of avg-dir energy <100Hz: {elow:.3f}")