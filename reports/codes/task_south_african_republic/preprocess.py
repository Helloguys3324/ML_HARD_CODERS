"""Preprocess stadium speech: noise reduction + bandpass -> clean.wav."""
import numpy as np
import soundfile as sf
import noisereduce as nr
from scipy.signal import butter, sosfiltfilt

SRC = "/home/kozak/mllm/challenges/task_south_african_republic/data/voice_note.wav"
DST = "/home/kozak/mllm/challenges/task_south_african_republic/clean.wav"

def preprocess(src, dst, lowcut=300.0, highcut=3400.0, prop_decrease=0.97):
    audio, sr = sf.read(src, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # noise profile from clean leading noise
    noise = audio[: 2 * sr]
    red = nr.reduce_noise(
        y=audio, sr=sr, y_noise=noise,
        prop_decrease=prop_decrease, stationary=True,
    )
    # bandpass 300-3400 Hz (speech band)
    sos = butter(4, [lowcut, highcut], btype="band", fs=sr, output="sos")
    filt = sosfiltfilt(sos, red)
    # normalize
    filt = filt / (np.max(np.abs(filt)) + 1e-9) * 0.98
    sf.write(dst, filt.astype("float32"), sr)
    print(f"wrote {dst} dur={len(filt)/sr:.1f}s")

if __name__ == "__main__":
    preprocess(SRC, DST)