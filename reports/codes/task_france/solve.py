#!/usr/bin/env python
"""
task_france -- "Point par point" (Easy, text answer)

Systematic search for the word that "Le Pointilliste" says is hidden in the
Paris accommodation data.  This script reproduces the analyses of the second
solver pass (2026-11-14, ~90 min) and writes the diagnostic figures plus an
exploratory labels.csv.

Run:
    /home/kozak/mllm/py solve.py [--data CSV] [--out DIR]

Main findings (details in solution.md):
  * listing_id is uniform on [1e6, 1e7]  -> IDs were regenerated, no signal.
  * price is a mean of k integer nightly prices (cents = k/2, k/3, k/4, ...),
    with minimum_nights as the strongest denominator; a 1/11 family exists.
    Neither family traces glyphs on the map.
  * No spatial subset found by price / rating / reviews / availability /
    host-count / room-type / property-type / missingness forms letters.
  * The point process is a very clumpy (building-level) synthetic-ish process;
    no glyph-sized strokes or lattices were detected at 20 m - 3 km scales.
  * The only strong oddity is a group of 7 rows with
    calculated_host_listings_count == 225, six of which have 3-decimal
    coordinates (48.862, 2.309), etc.  Seven scattered dots cannot spell a
    word; treated as a generator fingerprint, not as the answer.

The flag is NOT recovered with certainty.  The candidate in answer.txt is the
thematic best guess (SEURAT); confidence LOW.
"""
import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "..", "..", "tasks", "task_france", "data", "paris_listings.csv")
DEFAULT_OUT = HERE

# local metric approximation around Paris
LAT0, LON0 = 48.8566, 2.3522
M_PER_DEG_LAT = 111320.0
M_PER_DEG_LON = 111320.0 * np.cos(np.radians(LAT0))


def xy_m(df):
    """(x, y) in metres from the Paris centre."""
    return np.c_[
        (df.longitude.values - LON0) * M_PER_DEG_LON,
        (df.latitude.values - LAT0) * M_PER_DEG_LAT,
    ]


def audit(df):
    print("=" * 70)
    print("1. DATA AUDIT")
    print("shape:", df.shape)
    print("missing:\n", df.isna().sum()[df.isna().sum() > 0])

    ids = np.sort(df.listing_id.values)
    print("\nlisting_id range:", ids[0], "-", ids[-1])
    print("first digit counts:", np.bincount((ids // 10 ** 6).astype(int), minlength=10))
    print("last digit counts :", np.bincount(ids % 10, minlength=10))
    print("-> uniform 7-digit IDs; IDs carry no information (regenerated).")

    # coordinate precision
    def ndec(s):
        return s.astype(str).str.split(".").str[1].str.len().fillna(0).astype(int)

    print("\ncoordinate decimal places (lat):", ndec(df.latitude).value_counts().sort_index().to_dict())
    print("coordinate decimal places (lon):", ndec(df.longitude).value_counts().sort_index().to_dict())
    print("expected for uniform rounding to 5 dp: 90%/9%/0.9%/0.09%")
    both3 = (ndec(df.latitude) <= 3) & (ndec(df.longitude) <= 3)
    print("rows with both coords <= 3 dp:", int(both3.sum()), "(chance ~3)")
    print(df.loc[both3, ["listing_id", "neighbourhood", "latitude", "longitude",
                         "price", "minimum_nights", "availability_365",
                         "calculated_host_listings_count"]].to_string())


def price_structure(df):
    print("\n" + "=" * 70)
    print("2. PRICE GENERATION STRUCTURE")
    cents = np.round(df.price.values * 100).astype(int) % 100
    vc = pd.Series(cents).value_counts().sort_values(ascending=False)
    print("top cents:", vc.head(15).to_dict())
    print("-> spikes at 0/50, 0/33/67, 0/25/50/75, 0/20/40/60/80: prices are")
    print("   rounded means of k integer nightly prices; minimum_nights is the")
    print("   dominant k for small k.")

    # k/11 family (hotel signature)
    m11 = np.isin(cents, [9, 18, 27, 36, 45, 55, 64, 73, 82, 91])
    print("\nnon-integer k/11 prices:", int(m11.sum()),
          "(Room in hotel rows:", int(((df.property_type == "Room in hotel") & m11).sum()), ")")

    # the 225 oddity
    g225 = df[df.calculated_host_listings_count == 225]
    print("\ncalculated_host_listings_count == 225:", len(g225), "rows")
    print(g225[["listing_id", "neighbourhood", "latitude", "longitude", "price",
                "minimum_nights", "availability_365", "review_scores_rating"]].to_string())
    print("-> 6/7 share 3-dp coordinates and minimum_nights=3; generator fingerprint,")
    print("   but the 7 dots are scattered over Paris and cannot form a word.")


def spatial_audit(df, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors

    X = xy_m(df)
    lon, lat = df.longitude.values, df.latitude.values

    # figures directory
    figs = os.path.join(out, "figs")
    os.makedirs(figs, exist_ok=True)

    # 2.1 all points
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.scatter(lon, lat, s=0.5, c="k", alpha=0.4, linewidths=0)
    ax.set_aspect(1 / np.cos(np.radians(48.86)))
    ax.set_title(f"all listings ({len(df)})")
    fig.savefig(os.path.join(figs, "01_all_points.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 2.2 greyscale density (the "Grey is pretty" reading)
    nb = 600
    H, xe, ye = np.histogram2d(lon, lat, bins=[nb, int(nb * 0.75)],
                               range=[[2.24, 2.46], [48.815, 48.905]])
    H = gaussian_filter(H.astype(float), 2.0)
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.imshow(np.log1p(H.T), origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]],
              cmap="gray_r", aspect="auto", interpolation="nearest")
    ax.set_aspect(1 / np.cos(np.radians(48.86)))
    ax.set_title("greyscale density (no letters visible)")
    fig.savefig(os.path.join(figs, "02_density_grey.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 2.3 local contrast (over/under density)
    from scipy.ndimage import uniform_filter
    H2 = gaussian_filter(H, 1.0)
    bg = uniform_filter(H2, size=25)
    ratio = np.log((H2 + 0.1) / (bg + 0.1))
    fig, ax = plt.subplots(figsize=(16, 14))
    vmin, vmax = np.percentile(ratio, [1, 99])
    ax.imshow(ratio.T, origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]],
              cmap="gray_r", aspect="auto", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_aspect(1 / np.cos(np.radians(48.86)))
    ax.set_title("local density contrast (no glyph strokes)")
    fig.savefig(os.path.join(figs, "03_density_contrast.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 2.4 all "special" row subsets on one contact sheet
    def ndec(s):
        return s.astype(str).str.split(".").str[1].str.len().fillna(0).astype(int)

    probes = {
        "beds NaN": df.beds.isna(),
        "minimum_nights NaN": df.minimum_nights.isna(),
        "bedrooms NaN": df.bedrooms.isna(),
        "room in hotel": df.property_type == "Room in hotel",
        "hotel k/11": (df.property_type == "Room in hotel")
        & np.isin(np.round(df.price * 100).astype(int) % 100,
                  [9, 18, 27, 36, 45, 55, 64, 73, 82, 91]),
        "shared room": df.room_type == "Shared room",
        "availability 0": df.availability_365 == 0,
        "reviews_ltm 0": df.number_of_reviews_ltm == 0,
        "superhost": df.host_is_superhost,
        "latitude <=3dp": ndec(df.latitude) <= 3,
        "longitude <=3dp": ndec(df.longitude) <= 3,
        "host_count 225": df.calculated_host_listings_count == 225,
        "price 1/11 family": np.isin(np.round(df.price * 100).astype(int) % 100,
                                     [9, 18, 27, 36, 45, 55, 64, 73, 82, 91]),
        "price < 80": df.price < 80,
        "price > 800": df.price > 800,
        "rating == 5.0": df.review_scores_rating == 5.0,
    }
    fig, axs = plt.subplots(4, 4, figsize=(24, 20))
    for ax, (name, m) in zip(axs.ravel(), probes.items()):
        ax.scatter(lon, lat, s=0.25, c="lightgrey", linewidths=0)
        ax.scatter(lon[m], lat[m], s=6, c="red", linewidths=0)
        ax.set_aspect(1 / np.cos(np.radians(48.86)))
        ax.set_title(f"{name}: {int(m.sum())}", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(figs, "04_probe_subsets.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)

    # 2.5 isolation / sparse points
    nn = NearestNeighbors(n_neighbors=201).fit(X)
    d, _ = nn.kneighbors(X)
    d200 = d[:, 200]
    r = np.hypot(X[:, 0], X[:, 1])
    inner = r < 5500
    thr = np.quantile(d200[inner], 0.99)
    m = inner & (d200 > thr)
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.scatter(lon[inner], lat[inner], s=0.25, c="lightgrey", linewidths=0)
    ax.scatter(lon[m], lat[m], s=8, c="red", linewidths=0)
    ax.set_aspect(1 / np.cos(np.radians(48.86)))
    ax.set_title(f"most isolated points (d200 > {thr:.0f} m): city-edge only")
    fig.savefig(os.path.join(figs, "05_isolated.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 2.6 DBSCAN components: look for compact letter-sized clusters
    lab = DBSCAN(eps=35, min_samples=3).fit_predict(X)
    u, c = np.unique(lab, return_counts=True)
    clusters = [(l, int(n)) for l, n in zip(u, c) if l != -1]
    clusters.sort(key=lambda a: -a[1])
    print("\nDBSCAN eps=35m min_samples=3: %d clusters; largest sizes %s"
          % (len(clusters), [n for _, n in clusters[:10]]))
    print("-> one giant city component; no isolated glyph-sized component.")

    # exploratory labels.csv (nearest spatial cluster at a larger eps)
    lab200 = DBSCAN(eps=200, min_samples=5).fit_predict(X)
    pd.DataFrame({"listing_id": df.listing_id, "dbscan_cluster": lab200}).to_csv(
        os.path.join(out, "labels.csv"), index=False)
    print("wrote exploratory labels.csv (DBSCAN eps=200 m, min_samples=5)")

    # 2.7 nearest-neighbour direction / distance (regular-stroke test)
    nn4 = NearestNeighbors(n_neighbors=4).fit(X)
    d4, idx4 = nn4.kneighbors(X)
    angles, dists = [], []
    for k in (1, 2, 3):
        P = X[idx4[:, k]] - X
        a = np.degrees(np.arctan2(P[:, 1], P[:, 0])) % 180
        good = d4[:, k] > 2
        angles.append(a[good])
        dists.append(d4[good, k])
    angles = np.concatenate(angles)
    dists = np.concatenate(dists)
    h, e = np.histogram(angles, bins=72, range=(0, 180))
    print("\nNN direction histogram peak at %.1f deg (%d vs mean %.0f): "
          "explained by 1e-5 deg coordinate rounding" % (e[h.argmax()], h.max(), h.mean()))
    hd, ed = np.histogram(dists, bins=np.arange(0, 120, 1))
    print("NN distance histogram: spike at 0 m (duplicate coords), then broad 8-16 m;"
          " no glyph-pitch spike.")

    # 2.8 lattice / phase scan (degree and metre grids)
    print("\ngrid-phase scan (max 2-D phase bin vs expected):")
    for p in (0.005, 0.002, 0.001, 0.0005, 0.0002):
        fx = (lon / p) % 1.0
        fy = (lat / p) % 1.0
        Hp, _, _ = np.histogram2d(fx, fy, bins=[40, 40], range=[[0, 1], [0, 1]])
        print("  p=%.4f deg: max bin %3d (expected %.1f)" % (p, Hp.max(), len(df) / 1600))
    print("-> no lattice; no axis-aligned pixel-font grid.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.data)
    audit(df)
    price_structure(df)
    spatial_audit(df, args.out)

    print("\n" + "=" * 70)
    print("CONCLUSION: the word was not recovered from the data.")
    print("The answer candidate in answer.txt is a thematic best guess (SEURAT);")
    print("confidence LOW.  Do not submit without an independent confirmation.")


if __name__ == "__main__":
    main()
