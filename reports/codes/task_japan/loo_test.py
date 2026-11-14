import numpy as np, glob, os
from PIL import Image
from scipy import ndimage

ROOT = '/home/kozak/mllm/MML/work/tasks/task_japan/data/dataset'

def load_gray(path):
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.float32).mean(axis=2)

def residual(g):
    return g - ndimage.gaussian_filter(g, 1.2)

def ncc(x, y):
    x = x.ravel() - x.mean(); y = y.ravel() - y.mean()
    return float((x*y).mean()/(x.std()*y.std()+1e-12))

d = np.load('/home/kozak/mllm/MML/work/solutions/task_japan/fps.npz')
fps = {k: d[k] for k in d.files}
handles = list(fps)

def main():
    print('Per-image NCC against each person fingerprint (own vs others):')
    mat = []
    for h in handles:
        paths = sorted(glob.glob(os.path.join(ROOT, 'credited', h, '*.jpg')))
        for p in paths:
            g = load_gray(p)
            r = residual(g)[40:850, 40:1240]
            scores = {hh: ncc(r, fps[hh]) for hh in handles}
            mat.append((h, os.path.basename(p), scores))
    # own score vs best other
    own = []
    for h, name, sc in mat:
        own.append((sc[h], max(v for k, v in sc.items() if k != h)))
    own = np.array(own)
    print('mean own NCC %.5f  mean best-other %.5f  diff %.5f' % (own[:,0].mean(), own[:,1].mean(), (own[:,0]-own[:,1]).mean()))
    # accuracy
    correct = 0
    for h, name, sc in mat:
        if max(sc, key=sc.get) == h: correct += 1
    print('nearest-fingerprint accuracy: %d/%d' % (correct, len(mat)))
    # per person separation
    for h in handles:
        vals = [(sc[h], max(v for k, v in sc.items() if k != h)) for hh, name, sc in mat if hh == h]
        v = np.array(vals)
        print('%12s own %.5f others %.5f gap %.5f' % (h, v[:,0].mean(), v[:,1].mean(), (v[:,0]-v[:,1]).mean()))
    np.save('/home/kozak/mllm/MML/work/solutions/task_japan/leaveoneout.npy',
            np.array([[sc[h] for h in handles] for _,_,sc in mat]))
    with open('/home/kozak/mllm/MML/work/solutions/task_japan/loo_names.txt','w') as f:
        for h, name, sc in mat:
            f.write(h + '/' + name + '\n')

if __name__ == '__main__':
    main()
