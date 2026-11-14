import pandas as pd, numpy as np
from scipy import ndimage

df = pd.read_csv('challenges/task_france/data/paris_listings.csv')
lon = df['longitude'].values
lat = df['latitude'].values
N = len(df)

def components_score(mask, W=300, H=150, auto=True):
    m = np.asarray(mask)
    n = m.sum()
    if n < 30: return None, n
    if auto:
        xlim = (lon[m].min(), lon[m].max()); ylim = (lat[m].min(), lat[m].max())
    else:
        xlim = (lon.min(), lon.max()); ylim = (lat.min(), lat.max())
    xb = np.linspace(xlim[0], xlim[1], W+1)
    yb = np.linspace(ylim[0], ylim[1], H+1)
    h,_,_ = np.histogram2d(lon[m], lat[m], bins=[xb,yb])
    h = (h.T > 0)
    # dilate to connect near dots
    h = ndimage.binary_dilation(h, iterations=1)
    lbl, nc = ndimage.label(h)
    if nc == 0: return None, n
    sizes = ndimage.sum(h, lbl, range(1, nc+1))
    # points per component, weighted
    return sizes, n

def describe(mask, name):
    sizes, n = components_score(mask)
    if sizes is None:
        return None
    return (sizes, n, name)

def top_components(mask, name):
    sizes, n, _ = describe(mask, name)
    return sizes

# ---- battery of filters ----
cands = []
def add(mask, name):
    if mask.sum() < 30 or mask.sum() > 1000:
        return
    s, n = components_score(mask)
    if s is None: return
    # score = avg points per component (letters => few big components)
    score = n / len(s)
    cands.append((score, n, len(s), name, mask))

# numeric columns
numcols = ['price','accommodates','bedrooms','beds','minimum_nights','availability_365',
           'number_of_reviews','number_of_reviews_ltm','reviews_per_month',
           'review_scores_rating','calculated_host_listings_count']
for c in numcols:
    v = df[c].rank(pct=True)
    for i in range(20):  # 5% quantile bands
        lo = i*0.05; hi = lo+0.05
        m = (v>=lo)&(v<hi)
        add(m, f'{c} decile {i}')

# derived value metrics
v1 = (df['review_scores_rating']/df['price']).rank(pct=True)
v2 = (df['price']/df['review_scores_rating']).rank(pct=True)
v3 = (df['review_scores_rating'].rank(pct=True).values - df['price'].rank(pct=True).values)
for name, vv in [('rating/price', v1), ('price/rating', v2)]:
    for i in range(20):
        lo=i*0.05; hi=lo+0.05
        m=(vv>=lo)&(vv<hi)
        add(m, f'{name} decile {i}')
# rank diff bands
pr = df['price'].rank(pct=True).values
rr = df['review_scores_rating'].rank(pct=True).values
d = rr - pr
for i in range(20):
    lo = -1+i*0.1; hi = lo+0.1
    m = (d>=lo)&(d<hi)
    add(m, f'rankdiff band {i}')

cands.sort(reverse=True)
print('TOP 40 candidates by clustering score:')
for score, n, nc, name, mask in cands[:40]:
    print(f'score={score:.2f} n={n} comps={nc} | {name}')