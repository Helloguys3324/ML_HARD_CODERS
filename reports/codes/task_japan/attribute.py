import numpy as np, glob, os, csv, collections, json
from PIL import Image
from scipy import ndimage

ROOT = '/home/kozak/mllm/MML/work/tasks/task_japan/data/dataset'
OUT = '/home/kozak/mllm/MML/work/solutions/task_japan'

def load_gray(path):
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.float32).mean(axis=2)

def residual(g):
    return g - ndimage.gaussian_filter(g, 1.2)

def ncc(x, y):
    x = x.ravel() - x.mean(); y = y.ravel() - y.mean()
    return float((x*y).mean()/(x.std()*y.std()+1e-12))

def main():
    d = np.load(os.path.join(OUT, 'fps.npz'))
    fps = {k: d[k] for k in d.files}
    handles = sorted(fps)
    anon = sorted(glob.glob(os.path.join(ROOT, 'anonymous', '*.jpg')))
    rows = []
    for p in anon:
        g = load_gray(p)
        r = residual(g)[40:850, 40:1240]
        scores = {h: ncc(r, fps[h]) for h in handles}
        best = max(scores, key=scores.get)
        srt = sorted(scores.values(), reverse=True)
        rows.append((os.path.basename(p), best, scores, srt[0]-srt[1]))
    print('counts:', dict(collections.Counter(r[1] for r in rows)))
    print('score distribution by assigned:')
    for h in handles:
        v = np.array([r[2][h] for r in rows if r[1] == h])
        if len(v): print(' %12s n=%3d mean %.4f min %.4f max %.4f' % (h, len(v), v.mean(), v.min(), v.max()))
    # low-confidence
    lows = sorted(rows, key=lambda r: r[3])[:15]
    print('\nlowest margin:')
    for name, best, sc, marg in lows:
        print(' %s -> %-12s marg %.4f  %s' % (name, best, marg, ' '.join('%s:%.3f'%(h,sc[h]) for h in handles)))
    with open(os.path.join(OUT, 'attribution.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['file', 'week', 'handle', 'margin'] + handles)
        for name, best, sc, marg in rows:
            w.writerow([name, int(name[1:3]), best, '%.4f' % marg] + ['%.5f' % sc[h] for h in handles])
    # per-week distribution
    print('\nper-week attribution:')
    tab = collections.defaultdict(collections.Counter)
    for name, best, sc, marg in rows:
        tab[int(name[1:3])][best] += 1
    for wk in sorted(tab):
        print(' w%02d total %2d : %s' % (wk, sum(tab[wk].values()), dict(tab[wk])))
    np.save(os.path.join(OUT, 'anon_scores.npy'), np.array([[r[2][h] for h in handles] for r in rows]))
    with open(os.path.join(OUT, 'anon_names.txt'), 'w') as f:
        for name, best, sc, marg in rows: f.write(name + '\n')

if __name__ == '__main__':
    main()
