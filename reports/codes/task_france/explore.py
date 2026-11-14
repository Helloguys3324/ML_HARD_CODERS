import pandas as pd, numpy as np

df = pd.read_csv('challenges/task_france/data/paris_listings.csv')
lon = df['longitude'].values
lat = df['latitude'].values

def show(mask, name, W=200, H=90, xlim=None, ylim=None, force_full=False):
    m = np.asarray(mask)
    n = m.sum()
    if xlim is None:
        if force_full:
            xlim = (lon.min(), lon.max())
        else:
            xlim = (lon[m].min(), lon[m].max())
    if ylim is None:
        if force_full:
            ylim = (lat.min(), lat.max())
        else:
            ylim = (lat[m].min(), lat[m].max())
    xb = np.linspace(xlim[0], xlim[1], W+1)
    yb = np.linspace(ylim[0], ylim[1], H+1)
    h, _, _ = np.histogram2d(lon[m], lat[m], bins=[xb, yb])
    h = h.T
    # shade by density
    mx = h.max()
    chars = ' .:*#@'
    print(f'=== {name} n={n} ===')
    for r in range(H):
        row = ''
        for c in range(W):
            v = h[r, c]
            if v == 0:
                row += ' '
            else:
                idx = int(v / mx * (len(chars)-1))
                idx = min(idx, len(chars)-1)
                row += chars[idx]
        print(row)
    return h

def maxdiff(xs):
    m = np.zeros_like(xs[0])
    for i in range(len(xs)):
        for j in range(i+1, len(xs)):
            m = np.maximum(m, np.abs(xs[i]-xs[j]))
    return m