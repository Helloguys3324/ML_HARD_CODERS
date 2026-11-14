import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
import numpy as np, csv, time
from scipy.io import wavfile
import torch
torch.set_num_threads(8)
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
nonrain = np.where(true != 3)[0]

def apply_dir(delta):
    """delta: raw (N,) -> apply per-sample energy scaling, return predictions."""
    with torch.no_grad():
        d = torch.from_numpy(delta.astype(np.float32))
        dn = d / (d.norm() + 1e-12)
        xp = x + C * xn * dn
        return m(xp).argmax(-1).numpy()

def pgd_direction(x_i, init_dir, steps=60, lr=0.05):
    """Per-sample PGD from the current universal direction; returns RAW delta (N,)."""
    xi = torch.tensor(audio[x_i][None])
    xi_n = xi.norm()
    base = C * xi_n  # full-budget magnitude
    d_init = torch.tensor(init_dir.astype(np.float32))
    d_init = base * d_init / (d_init.norm() + 1e-12)
    pert = d_init.clone().requires_grad_(True)
    for s in range(steps):
        out = m(xi + pert.reshape(1, N))
        loss = -out.log_softmax(-1)[:, TARGET].mean()
        g = torch.autograd.grad(loss, pert)[0]
        pert = pert - lr * g
        pn = pert.norm()
        if pn > base:
            pert = pert / pn * base
        pert = pert.detach().clone().requires_grad_(True)
    with torch.no_grad():
        r = pert.reshape(1, N).numpy().astype(np.float32)
    return r.reshape(N)

def evald(delta):
    p = apply_dir(delta)
    fl = []
    for c in range(4):
        mm = true == c
        fl.append(f"{int((p[mm] == 3).sum())}/{int(mm.sum())}")
    return p, fl

# start from the 19/23 UAP
try:
    delta = np.load("uap_rain_local.npy")
    print("loaded uap_rain_local.npy as init", flush=True)
except FileNotFoundError:
    delta = np.zeros(N, dtype=np.float32)
    rng = np.random.default_rng(0); delta = rng.standard_normal(N).astype(np.float32) * 1e-3

delta = delta / (np.linalg.norm(delta) + 1e-12)

p, fl = evald(delta)
print("init:", fl, flush=True)

best = delta.copy(); best_score = -1
lr_accum = 0.8
for epoch in range(8):
    p, fl = evald(delta)
    unflipped = [i for i in nonrain if p[i] != 3]
    n_un = len(unflipped)
    print(f"epoch {epoch}: unflipped={unflipped}", flush=True)
    if n_un == 0:
        break
    # accumulate per-sample PGD directions for unflipped samples
    accum = np.zeros(N, dtype=np.float32)
    for i in unflipped:
        r = pgd_direction(i, delta, steps=60)
        r = r / (np.linalg.norm(r) + 1e-12)
        accum += r
    accum /= max(n_un, 1)
    new = delta + lr_accum * accum
    new = new / (np.linalg.norm(new) + 1e-12)
    # evaluate new vs old; also track that rain stays rain
    p_new, fl_new = evald(new)
    score_new = int(((p_new != 3) | True).sum())  # placeholder
    # accept if it doesn't reduce flip count
    fl_new_total = int(((p_new[true != 3]) == 3).sum())
    fl_old_total = int(((p[true != 3]) == 3).sum())
    rain_new = int(((p_new[true == 3]) == 3).sum())
    rain_old = int(((p[true == 3]) == 3).sum())
    if (fl_new_total + rain_new) >= (fl_old_total + rain_old):
        delta = new
        print(f"  accepted -> {fl_new} rain_keep {rain_new}", flush=True)
    else:
        lr_accum *= 0.5
        print(f"  rejected (would lose flips), lr->{lr_accum:.3f}", flush=True)
    # track best
    p_b, fl_b = evald(delta)
    sc = int((p_b == 3).sum())
    if sc > best_score:
        best_score = sc
        best = delta.copy()
        print(f"  new best {sc}/31: {fl_b}", flush=True)

np.save("uap_rain_sgduap.npy", best.astype(np.float32))
p, fl = evald(best)
print("FINAL SGD-UAP:", fl, "best_score", best_score, flush=True)