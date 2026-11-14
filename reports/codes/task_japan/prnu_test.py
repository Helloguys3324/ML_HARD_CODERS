import numpy as np, glob, os, sys, json, time
from PIL import Image
from scipy import ndimage

ROOT = '/home/kozak/mllm/MML/work/tasks/task_japan/data/dataset'

def load(path):
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.float32)

def residual(g):
    # denoise with gaussian, high-pass
    return g - ndimage.gaussian_filter(g, 1.2)

def main():
    credited = {}
    for d in sorted(glob.glob(os.path.join(ROOT, 'credited', '*'))):
        credited[os.path.basename(d)] = sorted(glob.glob(d + '/*.jpg'))
    anon = sorted(glob.glob(os.path.join(ROOT, 'anonymous', '*.jpg')))

    # per-person average residual + per image noise std
    fps = {}
    stats = {}
    for h, paths in credited.items():
        acc = None
        vals = []
        for p in paths:
            rgb = load(p)
            g = rgb.mean(axis=2)
            r = residual(g)
            # crop out border and caption area to avoid text edges
            r = r[40:850, 40:1240]
            acc = r if acc is None else acc + r
            vals.append((p, float(r.std()), float(g.mean())))
        fps[h] = acc / len(paths)
        stats[h] = vals
        print(h, 'mean resid std per img', np.mean([v[1] for v in vals]), 'brightness', np.mean([v[2] for v in vals]))

    handles = list(credited)
    print('\nPerson fingerprint cross-correlation:')
    for i, a in enumerate(handles):
        row = []
        for b in handles:
            x = fps[a].ravel(); y = fps[b].ravel()
            x = x - x.mean(); y = y - y.mean()
            row.append((x*y).mean() / (x.std()*y.std()+1e-12))
        print(a, ' '.join(f'{v:+.4f}' for v in row))

    np.savez('/home/kozak/mllm/MML/work/solutions/task_japan/fps.npz',
             **{h: fps[h] for h in handles})
    json.dump({h: [[os.path.basename(p), s, m] for p, s, m in v] for h, v in stats.items()},
              open('/home/kozak/mllm/MML/work/solutions/task_japan/basic_stats.json','w'), indent=1)

if __name__ == '__main__':
    main()
