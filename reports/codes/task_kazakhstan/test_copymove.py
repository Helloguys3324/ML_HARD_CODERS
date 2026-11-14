import cv2, numpy as np

def copy_move(img, bsize=24, stride=6, th=0.90, min_dist=None):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if min_dist is None:
        min_dist = bsize
    h, w = g.shape
    xs = np.arange(0, w - bsize + 1, stride)
    ys = np.arange(0, h - bsize + 1, stride)
    pos = []
    bl = []
    for y in ys:
        for x in xs:
            b = g[y:y + bsize, x:x + bsize]
            b = b - b.mean()
            n = np.linalg.norm(b)
            b = np.zeros_like(b) if n < 1e-6 else b / n
            bl.append(b.flatten())
            pos.append((x, y))
    B = np.array(bl)
    pos = np.array(pos)
    sim = B @ B.T
    np.fill_diagonal(sim, -1)
    N = len(B)
    pairs = []
    for i in range(N):
        js = np.where(sim[i] > th)[0]
        for j in js:
            if j <= i:
                continue
            xi, yi = pos[i]
            xj, yj = pos[j]
            if max(abs(xi - xj), abs(yi - yj)) < min_dist:
                continue
            pairs.append((i, j, sim[i, j], xi, yi, xj, yj))
    return pairs, pos, g.shape

for n in [4, 6, 8, 10, 1, 2]:
    img = cv2.imread(f'challenges/task_kazakhstan/data/exhibits/exhibit_{n:02d}.jpg')
    mask = cv2.imread(f'challenges/task_kazakhstan/data/exhibits/exhibit_{n:02d}_mask.png', 0)
    ys, xs = np.where(mask > 128)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    pairs, pos, shp = copy_move(img)
    pairs.sort(key=lambda p: -p[2])
    print(f'exhibit_{n:02d} mask box ({x0},{y0})-({x1},{y1}); high-corr pairs={len(pairs)}')
    for p in pairs[:6]:
        i, j, c, xi, yi, xj, yj = p
        print(f'   corr={c:.3f} ({xi},{yi})<->({xj},{yj})')