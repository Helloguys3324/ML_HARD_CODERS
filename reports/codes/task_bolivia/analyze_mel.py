import numpy as np, csv
from scipy.io import wavfile
import torch
from challenge_model import load_model, predict_probabilities

m = load_model()
torch.set_num_threads(8)
rows = list(csv.DictReader(open('data/recordings.csv')))
label_map = {'birds':0,'frogs':1,'insects':2,'rain':3}
clips=[]
for r in rows:
    rate,smp=wavfile.read('data/'+r['path']); clips.append(smp.astype(np.float32)/32768.0)
audio=np.stack(clips)
true=np.array([label_map[r['label']] for r in rows])
pred = predict_probabilities(m,audio).argmax(-1)
N=32000

# mel via torch to match model (hann 512, hop 160, 257 bins)
with torch.no_grad():
    x=torch.from_numpy(audio)
    p=torch.stft(x, n_fft=512, hop_length=160, window=torch.hann_window(512), return_complex=True, center=True).abs().square()*4
    # build mel bank same as model
    mel_bank = m.semantic.mel_bank  # (257,64)
    mel = 10*torch.log10(torch.clamp(p.transpose(1,2) @ mel_bank, min=1e-10))
mel = mel.numpy()  # (B, T, 64)

# time-average log-mel
tm = mel.mean(1)  # (B,64)

def show(title, idx_success, idx_fail):
    a = tm[idx_fail].mean(0)
    b = tm[idx_success].mean(0)
    d = a-b
    print(f"\n{title}: {len(idx_fail)} fail vs {len(idx_success)} ref")
    print("  mel bins where fail>ref (log dB):")
    for j in np.argsort(d)[::-1][:12]:
        print(f"    melbin {j:2d}  diff={d[j]:+.2f}")

for c in range(3):
    fail=[i for i in range(len(audio)) if true[i]==c and pred[i]==3]
    ok=[i for i in range(len(audio)) if true[i]==c and pred[i]==c]
    show(f"class {['birds','frogs','insects'][c]}", ok, fail)

# rain vs non-rain correct
rain=[i for i in range(len(audio)) if true[i]==3]
nonrain_ok=[i for i in range(len(audio)) if true[i]!=3 and pred[i]==true[i]]
show("rain(8) vs nonrain-correct", nonrain_ok, rain)

# Now gradient toward rain on a CORRECT recording: what mel bins does it boost?
import torch.nn.functional as F
idx = 2  # correct birds
x = torch.tensor(audio[idx][None]).requires_grad_(True)
out = m(x)
g = torch.autograd.grad(out[:,3].sum(), x)[0]  # grad of rain logit
# perturbation direction = sign of grad * ... approximate. Compute energy by freq of grad
g = g.numpy()[0]
G = np.abs(np.fft.rfft(g))**2
f = np.fft.rfftfreq(N,1/16000)
for lo,hi in [(0,60),(60,200),(200,1000),(1000,3000),(3000,6000),(6000,8000)]:
    frac = G[(f>=lo)&(f<hi)].sum()/G.sum()
    print(f"  grad-of-rain energy {lo}-{hi}Hz: {frac:.3f}")