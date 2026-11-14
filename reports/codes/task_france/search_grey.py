import pandas as pd, numpy as np
from itertools import combinations

df = pd.read_csv('challenges/task_france/data/paris_listings.csv')
lon = df['longitude'].values
lat = df['latitude'].values

FEATS = ['price','accommodates','bedrooms','beds','minimum_nights','availability_365',
         'number_of_reviews','number_of_reviews_ltm','reviews_per_month',
         'review_scores_rating','calculated_host_listings_count']

def norm(vals, mode):
    if mode == 'rank':
        return pd.Series(vals).rank(pct=True).values
    else:  # minmax
        v = np.asarray(vals, float)
        v = (v - np.nanmin(v))
        r = np.nanmax(v) - np.nanmin(v)
        if r == 0: return np.zeros_like(v)
        return v/r

def cluster_index(mask, ncell=20):
    """Variance/mean of cell counts over bbox -> >1 clustered."""
    m = np.asarray(mask)
    if m.sum() < 20: return 1.0, m.sum()
    x = lon[m]; y = lat[m]
    xb = np.linspace(x.min(), x.max(), ncell+1)
    yb = np.linspace(y.min(), y.max(), ncell+1)
    h,_,_ = np.histogram2d(x, y, bins=[xb,yb])
    mu = h.mean()
    var = h.var()
    if mu == 0: return 1.0, m.sum()
    return var/mu, m.sum()

res = []
for triple in combinations(FEATS, 3):
    for mode in ['rank','minmax']:
        a = norm(df[triple[0]].values, mode)
        b = norm(df[triple[1]].values, mode)
        c = norm(df[triple[2]].values, mode)
        chroma = np.maximum.reduce([a,b,c]) - np.minimum.reduce([a,b,c])
        # select chroma near 0 (grey) -- bottom 1%
        thr = np.quantile(chroma, 0.01)
        mask = chroma <= thr
        ci, n = cluster_index(mask)
        res.append((ci, n, triple, mode))

res.sort(reverse=True)
print('TOP configs by cluster index (grey = low chroma, bottom 1%):')
for ci, n, triple, mode in res[:25]:
    print(f'ci={ci:.2f} n={n} mode={mode} {triple}')