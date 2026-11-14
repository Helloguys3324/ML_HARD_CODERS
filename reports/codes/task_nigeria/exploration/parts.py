import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUT = '/home/kozak/mllm/MML/work/solutions/task_nigeria'
G = np.load(os.path.join(OUT, 'message_matrix.npy'))
print(G.shape)
# G is (freq, time) whitened, time 0.55-9.85 s
H, W = G.shape
for i, (a, b, name) in enumerate([(0, W//3, 'part1'), (W//3, 2*W//3, 'part2'), (2*W//3, W, 'part3')]):
    plt.figure(figsize=(14, 4))
    plt.imshow(G[:, a:b], aspect='auto', cmap='gray_r', origin='lower')
    plt.title(name); plt.axis('off')
    plt.tight_layout(); plt.savefig(os.path.join(OUT, f'msg_{name}.png'), dpi=170)
    plt.close()
print('saved parts')
