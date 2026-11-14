import pandas as pd, numpy as np, cv2, subprocess, os, tempfile
from itertools import combinations

df = pd.read_csv('challenges/task_france/data/paris_listings.csv')
lon = df['longitude'].values
lat = df['latitude'].values
N = len(df)

TESS = ['tesseract', '{f}', 'stdout', '-l', 'afr', '--psm', '6', '-c', 'tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ']

def ocr_points(mask, W=900, H=400, R=2, k=5, it=1):
    m = np.asarray(mask)
    x = lon[m]; y = lat[m]
    if len(x) < 20: return ''
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    if xmax-xmin < 1e-9 or ymax-ymin < 1e-9: return ''
    # pad bbox a bit
    px = (x - xmin)/(xmax-xmin)
    py = (y - ymin)/(ymax-ymin)
    canvas = np.zeros((H, W), np.uint8)
    for xi, yi in zip(px, py):
        cx = int(xi*(W-1)); cy = int((1-yi)*(H-1))
        cv2.circle(canvas, (cx, cy), R, 255, -1)
    kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dil = cv2.dilate(canvas, kk, iterations=it)
    fd, fn = tempfile.mkstemp(suffix='.png')
    os.close(fd)
    cv2.imwrite(fn, dil)
    out = subprocess.run([a.format(f=fn) for a in TESS], capture_output=True, text=True).stdout.strip()
    os.remove(fn)
    # clean
    letters = ''.join(ch for ch in out.upper() if ch in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    return letters

def add_cand(cands, mask, name):
    n = mask.sum()
    if n < 30 or n > 1200: return
    w = ocr_points(mask)
    cands.append((w, n, name))

cands = []
numcols = ['price','accommodates','bedrooms','beds','minimum_nights','availability_365',
           'number_of_reviews','number_of_reviews_ltm','reviews_per_month',
           'review_scores_rating','calculated_host_listings_count']
for c in numcols:
    v = df[c].rank(pct=True)
    for i in range(20):
        m = (v>=i*0.05)&(v<(i+1)*0.05)
        add_cand(cands, m, f'{c}:dec{i}')

# value metrics
for nm, vv in [('rating/price', (df.review_scores_rating/df.price).rank(pct=True)),
               ('price/rating', (df.price/df.review_scores_rating).rank(pct=True))]:
    for i in range(20):
        m=(vv>=i*0.05)&(vv<(i+1)*0.05)
        add_cand(cands, m, f'{nm}:dec{i}')

# rank diff bands
pr = df.price.rank(pct=True).values; rr = df.review_scores_rating.rank(pct=True).values
d = rr - pr
for i in range(20):
    m=(d>=-1+i*0.1)&(d<-1+(i+1)*0.1)
    add_cand(cands, m, f'rankdiff:{i}')

# triples grey (low chroma)
def rankc(c): return df[c].rank(pct=True).values
for t in [('price','review_scores_rating','number_of_reviews'),
          ('price','review_scores_rating','reviews_per_month'),
          ('price','review_scores_rating','number_of_reviews_ltm'),
          ('price','review_scores_rating','availability_365'),
          ('price','review_scores_rating','calculated_host_listings_count'),
          ('review_scores_rating','number_of_reviews','reviews_per_month')]:
    a,b,c=[rankc(x) for x in t]
    chroma=np.maximum.reduce([a,b,c])-np.minimum.reduce([a,b,c])
    for pctlbl,pct in [('grey1',1.0),('grey0.3',0.3),('color99',99.0)]:
        thr=np.percentile(chroma,pct)
        m = chroma<=thr if pct<50 else chroma>=thr
        add_cand(cands, m, f'{t}:{pctlbl}')

# categorical
for col in ['room_type','property_type','host_is_superhost']:
    for val in df[col].dropna().unique():
        m = df[col]==val
        add_cand(cands, m, f'{col}={val}')

# exact frequent price values
for p, cnt in df.price.value_counts().items():
    if cnt >= 40:
        add_cand(cands, df.price==p, f'price={p}')

# print results: any with a readable word (len>=3)
cands.sort(key=lambda x: -len(x[0]))
print(f'total candidates scanned: {len(cands)}')
print('--- candidates with >=4 letters OCR output ---')
for w,n,name in cands:
    if len(w)>=4:
        print(f'{w!r} n={n} | {name}')