import cv2, numpy as np, glob, warnings
warnings.filterwarnings('ignore')
from pathlib import Path

def copymove_vote(g, bsize=20, stride=5, th=0.90, min_sep=20):
    # downsample 2x
    g2 = cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    h, w = g2.shape
    bs = bsize // 2
    st = stride
    pieces = []
    pos = []
    for y in range(0, h - bs + 1, st):
        for x in range(0, w - bs + 1, st):
            b = g2[y:y + bs, x:x + bs].astype(np.float32)
            b = b - b.mean()
            n = np.linalg.norm(b)
            b = np.zeros_like(b) if n < 1e-6 else b / n
            pieces.append(b.flatten())
            pos.append((x, y))
    B = np.array(pieces)
    pos = np.array(pos)
    corr = B @ B.T
    np.fill_diagonal(corr, -1)
    N = len(B)
    # find translation-consistent pairs
    hists = {}
    pairs = []
    for i in range(N):
        js = np.where(corr[i] > th)[0]
        for j in js:
            if j <= i:
                continue
            xi, yi = pos[i]; xj, yj = pos[j]
            if max(abs(xi - xj), abs(yi - yj)) < min_sep // 2:
                continue
            dx, dy = xi - xj, yi - yj
            key = (dx, dy)
            hists.setdefault(key, []).append((i, j, xi, yi, xj, yj))
    # keep translation vectors with >=3 overlapping-ish pairs
    vote = np.zeros((h, w), np.float32)
    for key, plist in hists.items():
        if len(plist) < 3:
            continue
        # check compactness: spread of matched positions
        allpos = [(xi, yi) for _, _, xi, yi, _, _ in plist]
        ax = np.ptp([p[0] for p in allpos]); ay = np.ptp([p[1] for p in allpos])
        # mark
        for _, _, xi, yi, xj, yj in plist:
            vote[yi:yi + bs, xi:xi + bs] += 1
            vote[yj:yj + bs, xj:xj + bs] += 1
    # upsample back
    return cv2.resize(vote, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)

names = sorted(glob.glob('challenges/task_kazakhstan/data/exhibits/exhibit_*.jpg'))
hits = []
for gi, p in enumerate(names):
    img = cv2.imread(p)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.imread(p.replace('.jpg', '_mask.png'), 0)
    ys, xs = np.where(mask > 128)
    cy, cx = ys.mean(), xs.mean()
    vm = copymove_vote(g)
    if vm.max() < 1e-6:
        hits.append((Path(p).stem, None, 0))
        print(f'{Path(p).stem}: no copy-move votes')
        continue
    sm = cv2.GaussianBlur(vm, (0, 0), 8)
    ay, ax = np.unravel_index(np.argmax(sm), sm.shape)
    d = np.hypot(ay - cy, ax - cx)
    # overlap of top region with mask
    hits.append((Path(p).stem, d, vm.max()))
    print(f'{Path(p).stem}: cm_argmax dist={d:.0f}px  maxvotes={vm.max():.0f}')
d = np.array([h[1] for h in hits if h[1] is not None])
print(f'--- copy-move hit: median {np.median(d):.0f}px, <=40px {(d<=40).mean():.2%}')