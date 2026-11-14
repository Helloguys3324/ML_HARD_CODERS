import cv2, numpy as np, glob, warnings, csv
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from learn_detector import extract

DATA = Path('challenges/task_kazakhstan/data')
OUT = Path('challenges/task_kazakhstan')

def stamps_on(page):
    d = np.abs(page.astype(np.int16) - np.array([38, 36, 34])).max(2)
    m = cv2.morphologyEx((d > 30).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(m)
    return [(x, y, w, h) for x, y, w, h, area in stats[1:] if w > 120 and h > 120]


def main():
    names = sorted(glob.glob(str(DATA / 'exhibits/exhibit_*.jpg')))
    KEY = None
    # train on all exhibits
    Xt, yt = [], []
    for gi, p in enumerate(names):
        img = cv2.imread(p)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        maps = extract(g, img)
        if KEY is None:
            KEY = list(maps.keys())
        mask = cv2.imread(p.replace('.jpg', '_mask.png'), 0)
        F = np.vstack([maps[k].ravel() for k in KEY]).T
        m = (mask > 128).ravel()
        pos = F[m]
        neg_idx = np.where(~m)[0]
        rng = np.random.RandomState(0)
        neg_idx = rng.choice(neg_idx, size=min(len(pos) * 2, len(neg_idx)), replace=False)
        Xt.append(np.vstack([pos, F[neg_idx]]))
        yt.append(np.concatenate([np.ones(len(pos)), np.zeros(len(neg_idx))]))
    Xt = np.vstack(Xt); yt = np.concatenate(yt)
    scaler = StandardScaler().fit(Xt)
    clf = LogisticRegression(max_iter=800, class_weight='balanced').fit(scaler.transform(Xt), yt)
    print('trained', KEY)

    rows = []
    albums = sorted(p for p in (DATA / 'albums').iterdir() if p.is_dir())
    for ai, album in enumerate(albums):
        pts = []
        for f in sorted(album.glob('*.jpg')):
            page = cv2.imread(str(f))
            g = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
            maps = extract(g, page)
            F = np.vstack([maps[k].ravel() for k in KEY]).T
            prob = clf.predict_proba(scaler.transform(F))[:, 1].reshape(g.shape)
            prob = cv2.GaussianBlur(prob.astype(np.float32), (0, 0), 30)
            # mask to stamp interiors
            inside = np.zeros(prob.shape, bool)
            for x, y, w, h in stamps_on(page):
                inside[y + 12:y + h - 12, x + 12:x + w - 12] = True
            prob[~inside] = -1
            cy, cx = np.unravel_index(np.argmax(prob), prob.shape)
            pts.append((cx, cy))
            rows.append([album.name, f.stem, cx - 25, cy - 25, 50, 50])
        # render letter sheet
        sheet = np.zeros((1200, 1600, 3), np.uint8)
        for cx, cy in pts:
            cv2.circle(sheet, (cx, cy), 16, (0, 0, 255), -1)
        cv2.imwrite(str(OUT / f'letter_{album.name}.png'), sheet)
        # ascii preview (1 char = 8px, big)
        small = cv2.cvtColor(sheet, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(small, (1600 // 8, 1200 // 8), interpolation=cv2.INTER_AREA)
        art = '\n'.join(''.join('#' if v > 60 else '.' for v in row) for row in small)
        print(f'==== {album.name} ====')
        print(art)
    with open(str(OUT / 'boxes.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['album', 'page', 'x', 'y', 'w', 'h']); w.writerows(rows)
    print('wrote boxes.csv')


if __name__ == '__main__':
    main()