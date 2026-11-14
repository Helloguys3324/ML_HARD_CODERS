import wave, numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal

DATA = '/home/kozak/mllm/MML/work/tasks/task_nigeria/data/challenge.wav'
OUT = '/home/kozak/mllm/MML/work/solutions/task_nigeria'

w = wave.open(DATA, 'rb')
sr = w.getframerate()
x = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float64)
x -= x.mean()

# band-pass 3000-3700 Hz, extract amplitude envelope at fine resolution
sos = signal.butter(6, [3000, 3700], btype='band', fs=sr, output='sos')
y = signal.sosfiltfilt(sos, x)
env = np.abs(signal.hilbert(y))

# time resolution 2 ms
hop = int(sr*0.002)
n = len(env)//hop
E = env[:n*hop].reshape(n, hop).mean(axis=1)
# normalize / binarize
thr = np.percentile(E, 55)  # letters are high-energy fraction
B = (E > thr).astype(float)
print('frames', n, 'thr', thr, 'on fraction', B.mean())

# render as binary strip (stretch vertically)
plt.figure(figsize=(26, 2))
plt.imshow(B[None, :], aspect='auto', cmap='gray_r',
           extent=[0, n*hop/sr, 0, 1], interpolation='nearest')
plt.xlabel('s'); plt.title('band energy binarized 3.0-3.7 kHz (dark=letter)')
plt.yticks([])
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'message_binary.png'), dpi=160)

# per-tone frequency inside the band: track instantaneous freq to see if letters
# are drawn as frequency-varying strokes (like real spectrogram text)
# Build a "text image" from STFT in the band with fine time bins
nper = 512
nover = 456   # ~2.5ms hop
f, t, S = signal.spectrogram(x, fs=sr, nperseg=nper, noverlap=nover, window='hann')
Sdb = 10*np.log10(S + 1e-12)
m = (f >= 2800) & (f <= 3900)
img = Sdb[m]
# normalise each time column by its median outside message? use global percentile
vmin, vmax = np.percentile(img, [20, 99.5])
img2 = np.clip((img - vmin)/(vmax - vmin), 0, 1)
plt.figure(figsize=(28, 3.2))
plt.imshow(img2, aspect='auto', cmap='gray_r', origin='lower',
           extent=[t[0], t[-1], f[m][0], f[m][-1]], interpolation='bilinear')
plt.xlabel('s'); plt.ylabel('Hz'); plt.title('message band binarized')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'message_binary2.png'), dpi=160)

# print column energy profile coarse to see 11 letter blocks
prof = img2.mean(axis=0)
# smooth
k = 7
prof_s = np.convolve(prof, np.ones(k)/k, mode='same')
on = prof_s > 0.28
edges = np.flatnonzero(np.diff(on.astype(int)))
print('on/off edges at times:', [round(t[e],2) for e in edges])
