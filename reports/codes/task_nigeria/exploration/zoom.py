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

# high time resolution spectrogram, narrow band around the message
nper = 512
nover = 480
f, t, Sxx = signal.spectrogram(x, fs=sr, nperseg=nper, noverlap=nover, window='hann')
Sdb = 10*np.log10(Sxx + 1e-12)
mask = (f >= 2000) & (f <= 4500)
plt.figure(figsize=(24, 4))
plt.pcolormesh(t, f[mask], Sdb[mask], shading='auto', cmap='inferno')
plt.ylabel('Hz'); plt.xlabel('s'); plt.title('zoom 2-4.5 kHz')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'spectrogram_zoom.png'), dpi=150)

# full band, wider dynamic range view (maybe hidden weak text elsewhere)
f2, t2, Sxx2 = signal.spectrogram(x, fs=sr, nperseg=2048, noverlap=1900, window='hann')
Sdb2 = 10*np.log10(Sxx2 + 1e-12)
for flo, fhi, name in [(0, 1200, 'low'), (4000, 11025, 'high'), (0, 11025, 'full')]:
    m = (f2 >= flo) & (f2 <= fhi)
    plt.figure(figsize=(24, max(2, 8*(fhi-flo)/11025)))
    vmin, vmax = np.percentile(Sdb2[m], [15, 99.7])
    plt.pcolormesh(t2, f2[m], Sdb2[m], shading='auto', cmap='inferno', vmin=vmin, vmax=vmax)
    plt.ylabel('Hz'); plt.xlabel('s'); plt.title(f'{name} band')
    plt.tight_layout(); plt.savefig(os.path.join(OUT, f'spectrogram_{name}.png'), dpi=150)
    print('saved', name, 'vmin/vmax', vmin, vmax)

# envelope of the message band (3.1-3.6 kHz) -> rectified signal, check for extra modulation
sos = signal.butter(6, [3000, 3700], btype='band', fs=sr, output='sos')
y = signal.sosfiltfilt(sos, x)
env = np.abs(signal.hilbert(y))
env_s = signal.decimate(env, 100)
print('band env stats', env_s.mean(), env_s.std(), env_s.min(), env_s.max())
# sample envelope at fine resolution as an "image" of the message
# also produce spectrogram with very high time resolution to read letters crisply
nper = 256; nover = 240
f3, t3, S3 = signal.spectrogram(x, fs=sr, nperseg=nper, noverlap=nover, window='hann')
S3db = 10*np.log10(S3 + 1e-12)
m = (f3 >= 2800) & (f3 <= 3900)
plt.figure(figsize=(26, 3))
vmin, vmax = np.percentile(S3db[m], [30, 99.9])
plt.pcolormesh(t3, f3[m], S3db[m], shading='auto', cmap='gray', vmin=vmin, vmax=vmax)
plt.ylabel('Hz'); plt.xlabel('s'); plt.title('message band high time-res')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'spectrogram_message.png'), dpi=160)
print('saved message band')
