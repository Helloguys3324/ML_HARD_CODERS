import pandas as pd, numpy as np, sys

def render(mask, name, gridsize=(160,80), xlim=None, ylim=None):
    df = pd.read_csv('challenges/task_france/data/paris_listings.csv')
    lon = df['longitude'].values
    lat = df['latitude'].values
    m = np.asarray(mask)
    if xlim is None:
        xlim = (lon[m].min(), lon[m].max())
    if ylim is None:
        ylim = (lat[m].min(), lat[m].max())
    H, W = gridsize
    xbins = np.linspace(xlim[0], xlim[1], W+1)
    ybins = np.linspace(ylim[0], ylim[1], H+1)
    hist, _, _ = np.histogram2d(lon[m], lat[m], bins=[xbins, ybins])
    hist = hist.T  # rows = lat, cols = lon
    # ASCII: use density
    lines = []
    dens = hist > 0
    for r in range(H-1, -1, -1):  # top row first = high latitude
        row = ''.join('#' if dens[r,c] else (' ' if hist[r,c]==0 else '.') for c in range(W))
        lines.append(row)
    print(f'=== {name} n={m.sum()} lat[{ylim[0]:.4f},{ylim[1]:.4f}] lon[{xlim[0]:.4f},{xlim[1]:.4f}] ===')
    print('\n'.join(lines))