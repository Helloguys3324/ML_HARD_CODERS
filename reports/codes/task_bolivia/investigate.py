import numpy as np, csv
import torch
from challenge_model import load_model, predict_probabilities, CLASS_NAMES

model = load_model()
torch.set_num_threads(8)

rows = list(csv.DictReader(open('data/recordings.csv')))
label_map = {'birds':0,'frogs':1,'insects':2,'rain':3}

from scipy.io import wavfile
clips = []
for r in rows:
    rate, smp = wavfile.read('data/'+r['path'])
    clips.append(smp.astype(np.float32)/32768.0)
audio = np.stack(clips)  # (31, 32000)
true = np.array([label_map[r['label']] for r in rows])

probs = predict_probabilities(model, audio)
pred = probs.argmax(-1)
print("Baseline predictions:")
for i in range(len(rows)):
    print(f"  {i:2d} true={true[i]} pred={pred[i]} p={probs[i].max():.3f}")

# energies
energies = (audio**2).sum(-1)
print("\nEnergy range:", energies.min(), energies.max(), "RMS of energies", np.sqrt(energies).mean())
print("min abs:", np.abs(audio).max(-1).min(), "max abs:", np.abs(audio).max())

def report(name, mod):
    e = ((mod-audio)**2).sum(-1) / energies  # fraction
    p2 = predict_probabilities(model, mod)
    pr = p2.argmax(-1)
    # success = flips non-rain -> rain
    nonrain = true != 3
    flipped = (true != 3) & (pr == 3)
    # keep rain as rain
    rain_ok = (true == 3) & (pr == 3)
    print(f"\n[{name}] max_energy%={e.max()*100:.3f} mean={e.mean()*100:.4f}")
    print(f"  flipped non-rain->rain: {flipped.sum()}/{nonrain.sum()}")
    print(f"  rain stays rain: {rain_ok.sum()}/{(true==3).sum()}")
    # breakdown
    for c in range(4):
        mask = true==c
        if mask.sum()==0: continue
        ok = (pr[mask]==3).sum()
        print(f"  source {CLASS_NAMES[c]:8s} ->rain: {ok}/{mask.sum()}")
    return pr, p2

rng = np.random.default_rng(0)

# Hypothesis 1: white noise at 1% energy
def add_noise(x, frac=0.01, seed=0):
    r = np.random.default_rng(seed)
    n = r.standard_normal(x.shape).astype(np.float32)
    n = n / np.sqrt((n**2).sum(-1, keepdims=True))  # unit energy per row
    return x + np.sqrt(frac) * n * np.sqrt((x**2).sum(-1, keepdims=True))

report("white noise 1%", add_noise(audio))

# Hypothesis 2: DC offset
def add_dc(x, frac=0.01):
    dc = np.ones_like(x) * np.sqrt(frac * (x**2).sum(-1, keepdims=True)/x.shape[1])
    return x + dc
report("DC offset 1%", add_dc(audio))

# Hypothesis 3: narrowband tone sweep
t = np.arange(32000)/16000.0
for freq in [50, 100, 200, 400, 1000, 2000, 4000, 8000]:
    for phase in ['sin','cos']:
        sig = np.sin(2*np.pi*freq*t) if phase=='sin' else np.cos(2*np.pi*freq*t)
        sig = sig.astype(np.float32)
        sig = sig / np.sqrt((sig**2).sum())
        tone = np.sqrt(0.01) * sig * np.sqrt(energies[:,None])
        # scale to <=1% energy exact
        report(f"tone {freq}Hz {phase}", audio + tone)