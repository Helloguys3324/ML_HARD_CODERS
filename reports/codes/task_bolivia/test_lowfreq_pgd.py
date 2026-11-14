import numpy as np, csv
from scipy.io import wavfile
import torch
from challenge_model import load_model

torch.set_num_threads(8)
m=load_model()
rows=list(csv.DictReader(open('data/recordings.csv')))
label_map={'birds':0,'frogs':1,'insects':2,'rain':3}
clips=[]
for r in rows:
    rate,smp=wavfile.read('data/'+r['path']); clips.append(smp.astype(np.float32)/32768.0)
audio=np.stack(clips)
true=np.array([label_map[r['label']] for r in rows])
N=32000; sr=16000; TARGET=3
freqs=np.fft.rfftfreq(N,1/sr)

# low-freq restriction: perturb only bins with f < corner
def mask_for(corner):
    m=(freqs<corner).astype(np.float32)  # (161,)
    return m

def fft_restricted_perturb(x0, corner, steps=80, lr=1.0):
    # parametrize perturbation in frequency domain, restrict to low-freq bins
    x = torch.tensor(x0[None], dtype=torch.float32)
    # we'll optimize complex spectrum of the delta (low bins only)
    # simpler: optimize input x directly but project gradient to low-freq each step
    xt = x.clone().requires_grad_(True)
    xb = x
    budget = 0.1 * x.norm()
    mask = torch.tensor(mask_for(corner), dtype=torch.float32)
    for i in range(steps):
        out = m(xt)
        loss = -out.log_softmax(-1)[:,TARGET].mean()
        g = torch.autograd.grad(loss, xt)[0]
        # project gradient to low-freq: FFT, mask, iFFT
        G = torch.fft.rfft(g, dim=-1) * mask
        gl = torch.fft.irfft(G, n=N, dim=-1)
        xt = xt - lr * gl
        d = xt - xb
        dn = d.norm()
        if dn > budget:
            d = d/dn*budget
        xt = (xb + d).detach().clone().requires_grad_(True)
    with torch.no_grad():
        p = m(xt).softmax(-1).numpy()[0]
    return p

for corner in [150, 300, 1000, 8000]:
    res=[]
    for idx in [2,6,3,14,10,25]:
        p = fft_restricted_perturb(audio[idx], corner)
        ok = p.argmax()==3
        res.append(f"idx{idx}({['birds','frogs','insects','rain'][true[idx]]})->{'rain' if ok else ['birds','frogs','insects','rain'][p.argmax()]}")
    print(f"corner<{corner}Hz:", " ".join(res))