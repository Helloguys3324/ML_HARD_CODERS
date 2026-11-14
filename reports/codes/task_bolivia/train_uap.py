import numpy as np, csv
import torch
from scipy.io import wavfile
from challenge_model import load_model, CLASS_NAMES

model = load_model()
torch.set_num_threads(8)
torch.manual_seed(0)

rows = list(csv.DictReader(open('data/recordings.csv')))
label_map = {'birds':0,'frogs':1,'insects':2,'rain':3}
clips = []
for r in rows:
    rate, smp = wavfile.read('data/'+r['path'])
    clips.append(smp.astype(np.float32)/32768.0)
audio = np.stack(clips)  # (31, 32000)
true = np.array([label_map[r['label']] for r in rows])

x = torch.from_numpy(audio.copy())  # (31, 32000)
x_norm = x.norm(dim=-1, keepdim=True)  # per sample energy sqrt

TARGET = 3
alpha = 0.0999  # sqrt(0.01) slightly under to be safe
lamb = 0.1

delta = torch.zeros(1, 32000, requires_grad=True, dtype=torch.float32)
# init with small white noise
torch.manual_seed(123)
delta.data = torch.randn(1, 32000) * 1e-4

opt = torch.optim.Adam([delta], lr=0.05)

def evaluate(deltavec):
    with torch.no_grad():
        d = deltavec.reshape(1, 32000)
        dn = d / (d.norm(dim=-1, keepdim=True) + 1e-12)
        # per-sample scale: alpha * x_norm
        scaled = alpha * x_norm * dn  # (31,32000)
        xp = x + scaled
        out = model(xp)
        pr = out.softmax(-1)
        pred = pr.argmax(-1).numpy()
        return pred

for it in range(400):
    opt.zero_grad()
    d = delta.reshape(1, 32000)
    dn = d / (d.norm(dim=-1, keepdim=True) + 1e-12)
    scaled = alpha * x_norm * dn
    xp = x + scaled
    out = model(xp)
    logp = out.log_softmax(-1)
    loss = -logp[:, TARGET].mean()
    loss.backward()
    opt.step()
    if it % 20 == 0:
        pred = evaluate(delta.detach())
        n_nonrain = (true != 3).sum()
        flipped = ((true != 3) & (pred == 3)).sum()
        rain_ok = ((true == 3) & (pred == 3)).sum() / max(1,(true==3).sum())
        print(f"it {it:3d} loss={loss.item():.4f} flipped {flipped}/{n_nonrain} rain_ok={rain_ok:.2f}")

dl = delta.detach().numpy().reshape(32000).astype(np.float32)
np.save('uap_rain.npy', dl)
print("\nSaved uap_rain.npy, norm", np.linalg.norm(dl))