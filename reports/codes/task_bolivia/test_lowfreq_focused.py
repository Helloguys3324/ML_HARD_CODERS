import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
import numpy as np, csv
from scipy.io import wavfile
import torch
torch.set_num_threads(8)
import sys
sys.path.insert(0, "data")
from challenge_model import load_model

m = load_model()
rows = list(csv.DictReader(open("data/recordings.csv")))
label_map = {"birds": 0, "frogs": 1, "insects": 2, "rain": 3}
clips = []
for r in rows:
    rate, smp = wavfile.read("data/" + r["path"])
    clips.append(smp.astype(np.float32) / 32768.0)
audio = np.stack(clips)
true = np.array([label_map[r["label"]] for r in rows])
N = 32000; TARGET = 3
freqs = torch.fft.rfftfreq(N, 1 / 16000)
NMS = ["birds", "frogs", "insects", "rain"]

def pgd_restricted(x0, corner, steps=50, lr=3.0):
    mask = torch.tensor((freqs < corner).float())
    xt = torch.tensor(x0[None]).clone().requires_grad_(True)
    xb = torch.tensor(x0[None])
    budget = 0.1 * xb.norm()
    for i in range(steps):
        loss = -(m(xt).log_softmax(-1)[:, TARGET].mean())
        g = torch.autograd.grad(loss, xt)[0]
        G = torch.fft.rfft(g) * mask
        gl = torch.fft.irfft(G, n=N)
        xt = xt - lr * gl
        d = xt - xb; dn = d.norm()
        if dn > budget:
            d = d / dn * budget
        xt = (xb + d).detach().clone().requires_grad_(True)
    with torch.no_grad():
        p = m(xt).softmax(-1).numpy()[0]
    return p

for corner in [150, 400, 8000]:
    out = []
    for idx in [10, 25, 6, 14, 29]:
        p = pgd_restricted(audio[idx], corner, steps=50)
        out.append(f"{NMS[true[idx]]}->{NMS[p.argmax()]}(p3={p[3]:.2f})")
    print(f"corner<{corner}: ", " | ".join(out), flush=True)