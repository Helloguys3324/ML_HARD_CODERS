import numpy as np, csv
from scipy.io import wavfile
import torch
torch.set_num_threads(4)
import sys
sys.path.insert(0, "data")
from challenge_model import load_model, predict_probabilities

m = load_model()
rows = list(csv.DictReader(open("data/recordings.csv")))
label_map = {"birds": 0, "frogs": 1, "insects": 2, "rain": 3}
clips = []
for r in rows:
    rate, smp = wavfile.read("data/" + r["path"])
    clips.append(smp.astype(np.float32) / 32768.0)
audio = np.stack(clips)
true = np.array([label_map[r["label"]] for r in rows])
pred = predict_probabilities(m, audio).argmax(-1)
N = 32000; sr = 16000
f = np.fft.rfftfreq(N, 1 / sr)

# rain-flip failures (non-rain -> rain) vs references (correct)
fail = np.array([i for i in range(len(audio)) if true[i] != 3 and pred[i] == 3])
ref = np.array([i for i in range(len(audio)) if pred[i] == true[i] and true[i] != 3])
print("failures:", fail, "refs:", ref)

S = np.abs(np.fft.rfft(audio, axis=-1))  # magnitude
# normalize each recording's spectrum so we compare SHAPE not loudness
def normspec(idx):
    m = S[idx]
    return m / (m.sum(-1, keepdims=True) + 1e-12)

sf = normspec(fail).mean(0)
sr_ = normspec(ref).mean(0)
df = sf - sr_
# find narrowband peaks where failures > refs, cross-class
order = np.argsort(df)[::-1]
print("\ntop freqs where ALL failures (pooled) exceed refs:")
for j in order[:15]:
    print(f"  f={f[j]:7.1f} Hz  diff={df[j]:+.5f}  fail={sf[j]:.5f} ref={sr_[j]:.5f}")

# coherence: is there a single freq present in EVERY failure with high rel magnitude?
print("\nper-failure top-3 freqs (relative magnitude):")
for i in fail:
    m = normspec([i])[0]
    top = np.argsort(m)[::-1][:3]
    print(f"  idx{i:2d} {['birds','frogs','insects','rain'][true[i]]:8s}: " + ", ".join(f"{f[j]:.0f}Hz" for j in top))