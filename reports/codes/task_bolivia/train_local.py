import os, time
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
import numpy as np, csv
from scipy.io import wavfile
import torch
torch.set_num_threads(4)
import sys
sys.path.insert(0, "data")
from challenge_model import load_model

m = load_model()
torch.manual_seed(0)
rows = list(csv.DictReader(open("data/recordings.csv")))
label_map = {"birds": 0, "frogs": 1, "insects": 2, "rain": 3}
clips = []
for r in rows:
    rate, smp = wavfile.read("data/" + r["path"])
    clips.append(smp.astype(np.float32) / 32768.0)
audio = np.stack(clips)
true = np.array([label_map[r["label"]] for r in rows])
N = 32000; TARGET = 3; C = 0.0999

x = torch.from_numpy(audio.copy())
xn = x.norm(dim=-1, keepdim=True)

def rand_vec(seed):
    torch.manual_seed(seed)
    return torch.randn(N) * 1e-3

delta = torch.zeros(N, requires_grad=True)
delta.data = rand_vec(123)

opt = torch.optim.Adam([delta], lr=0.4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)

def evaldelta():
    with torch.no_grad():
        d = delta.reshape(1, N); dn = d / (d.norm() + 1e-12)
        xp = x + C * xn * dn
        return m(xp).argmax(-1).numpy()

best_flip = -1
best_delta = None
t0 = time.time()
for it in range(60):
    opt.zero_grad()
    d = delta.reshape(1, N); dn = d / (d.norm() + 1e-12)
    xp = x + C * xn * dn
    out = m(xp)
    lp = out.log_softmax(-1)[:, TARGET]
    pp = out.softmax(-1)
    w = (1 - pp[:, TARGET].detach()).clamp(min=0.02)
    loss = -(lp * w).sum() / w.sum()
    loss.backward()
    opt.step(); sched.step()
    if it % 5 == 0 or it == 59:
        pred = evaldelta()
        fl = int(((pred[true != 3]) == 3).sum())
        rok = int(((pred[true == 3]) == 3).sum())
        per = [f"{int((pred[true == c] == 3).sum())}/{int((true == c).sum())}" for c in range(4)]
        print(f"it {it:2d} t={time.time()-t0:5.0f}s loss={loss.item():.4f} flip {fl}/23 rain {rok}/8 per={per}", flush=True)
        score = fl + rok
        if score > best_flip:
            best_flip = score
            best_delta = delta.detach().numpy().copy()

best_delta = best_delta if best_delta is not None else delta.detach().numpy()
best_delta = np.asarray(best_delta, dtype=np.float32)
dn = np.linalg.norm(best_delta); best_delta = best_delta / dn
np.save("uap_rain_local.npy", best_delta)
print("saved uap_rain_local.npy, best score", best_flip)