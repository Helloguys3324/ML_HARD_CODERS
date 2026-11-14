import pandas as pd, numpy as np

df = pd.read_csv('challenges/task_france/data/paris_listings.csv')
lon = df['longitude'].values
lat = df['latitude'].values

def local_map(vals, name, W=140, H=60, agg='mean'):
    xb = np.linspace(lon.min(), lon.max(), W+1)
    yb = np.linspace(lat.min(), lat.max(), H+1)
    sumv,_,_ = np.histogram2d(lon, lat, bins=[xb,yb], weights=vals)
    cnt,_,_ = np.histogram2d(lon, lat, bins=[xb,yb])
    with np.errstate(invalid='ignore', divide='ignore'):
        if agg=='mean':
            m = sumv/cnt
        elif agg=='sum':
            m = sumv
    m = m.T
    m[cnt.T==0] = np.nan
    # normalize and map to chars
    finite = m[np.isfinite(m)]
    lo, hi = np.nanpercentile(finite, 2), np.nanpercentile(finite, 98)
    chars = ' .:-=+*#%@'
    print(f'=== {name} (range {lo:.2f}..{hi:.2f}) ===')
    for r in range(H):
        row=''
        for c in range(W):
            v = m[r,c]
            if not np.isfinite(v):
                row+=' '
            else:
                idx = int((v-lo)/(hi-lo)*(len(chars)-1))
                idx=max(0,min(idx,len(chars)-1))
                row+=chars[idx]
        print(row)

if __name__=='__main__':
    local_map(df['review_scores_rating'].values, 'local mean rating')
    local_map(df['price'].values, 'local mean price')
    v = df['review_scores_rating'].values/df['price'].values
    local_map(v, 'local mean value(price/rating)')