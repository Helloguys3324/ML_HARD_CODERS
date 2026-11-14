import numpy as np, csv
from scipy.io import wavfile
from scipy import signal

rows = list(csv.DictReader(open('data/recordings.csv')))
label_map = {'birds':0,'frogs':1,'insects':2,'rain':3}
clips, meta = [], []
for r in rows:
    rate, smp = wavfile.read('data/'+r['path'])
    clips.append(smp.astype(np.float32)/32768.0)
    meta.append(r)
audio = np.stack(clips)
true = np.array([label_map[r['label']] for r in rows])

# failed -> rain (12) vs the one birds flips to rain too we treat separately
# identify the "rain-flip" failures (status misclassified & pred rain) via model quickly
import torch
from challenge_model import load_model, predict_probabilities
m = load_model()
pred = predict_probabilities(m, audio).argmax(-1)
rainflip = np.array([(true[i]!=3 and pred[i]==3) for i in range(len(audio))])
print("rain-flip failures:", rainflip.sum(), "of", (true!=3).sum(), "non-rain")

# power spectra
f = np.fft.rfftfreq(32000, 1/16000)
def magspec(x):
    X = np.fft.rfft(x, axis=-1)
    return np.abs(X)

specs = magspec(audio)
# average log spectral magnitude: failures(rainflip) vs references(same-class, correct)
def avg_log(xs):
    return np.log10(np.maximum(xs.mean(0), 1e-4))

groups = {}
for c in range(3):  # birds, frogs, insects
    ref = [i for i in range(len(audio)) if true[i]==c and pred[i]==c]
    fl = [i for i in range(len(audio)) if true[i]==c and rainflip[i]]
    if ref and fl:
        diff = avg_log(specs[fl]) - avg_log(specs[ref])
        print(f"\nclass={c}: {len(fl)} failures vs {len(ref)} refs")
        # top frequencies where failures have MORE energy
        idx = np.argsort(diff)[::-1][:15]
        for j in idx:
            print(f"  f={f[j]:7.1f} Hz  logdiff={diff[j]:+.3f}")

# Also: overall rain class reference spectrum to see what "rain" looks like
rain_ref = [i for i in range(len(audio)) if true[i]==3]
print("\nrain refs avg spectrum shape vs non-rain refs")
nr = [i for i in range(len(audio)) if true[i]!=3 and pred[i]==true[i]]
d = avg_log(specs[rain_ref]) - avg_log(specs[nr])
idx = np.argsort(d)[::-1][:20]
for j in idx:
    print(f"  f={f[j]:7.1f} Hz logdiff={d[j]:+.3f}")