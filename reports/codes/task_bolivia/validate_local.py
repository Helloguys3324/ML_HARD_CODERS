import sys
import numpy as np
from runpy import run_path

sys.path.insert(0, "data")
from scipy.io import wavfile
import csv
from audio_io import validate
from challenge_model import load_model, predict_probabilities, CLASS_NAMES

transform = run_path("main.py")["transform"]

# 31 recordings
rows = list(csv.DictReader(open("data/recordings.csv")))
label_map = {"birds": 0, "frogs": 1, "insects": 2, "rain": 3}
clips = []
for r in rows:
    rate, smp = wavfile.read("data/" + r["path"])
    clips.append(smp.astype(np.float32) / 32768.0)
audio = np.stack(clips)
true = np.array([label_map[r["label"]] for r in rows])

model = load_model()
modified = transform(audio.copy())
try:
    per = validate(audio, modified)
    print("validate OK, max energy %:", per.max())
except ValueError as e:
    print("VALIDATE FAIL:", e)
    per = None

before = predict_probabilities(model, audio).argmax(-1)
after = predict_probabilities(model, modified).argmax(-1)

flips = []
for c in range(4):
    m = true == c
    flips.append(f"{int((after[m] == 3).sum())}/{int(m.sum())}")
n_non = int((true != 3).sum())
print(f"flip nonrain->rain: {int(((true!=3)&(after==3)).sum())}/{n_non}")
print(f"rain stays rain: {int(((true==3)&(after==3)).sum())}/{int((true==3).sum())}")
print(f"per-class (birds,frogs,insects,rain) -> rain: {flips}")

print("\nper-recording: before -> after")
for i in range(len(audio)):
    print(f"  {i:2d} true={CLASS_NAMES[true[i]]:8s} {CLASS_NAMES[before[i]]:8s} -> {CLASS_NAMES[after[i]]:8s}  e={per[i]:.3f}%")

# also run on train-input.npz (formats from baseline)
with np.load("data/train-input.npz", allow_pickle=False) as d:
    orig = d["audio"]
    mod = transform(orig.copy())
    p_orig = validate(orig, mod)
    print("\ntrain-input energy max %:", p_orig.max())
    b = predict_probabilities(model, orig).argmax(-1)
    a = predict_probabilities(model, mod).argmax(-1)
    print("train-input before->after:", [CLASS_NAMES[x] for x in b], "->", [CLASS_NAMES[x] for x in a])