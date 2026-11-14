#!/usr/bin/env python3
"""task_laos "Bad BOMB" — unsupervised anomaly detection on shell images.

Finds 28 defective shells among 1225 (512x512 png) using an ensemble of
anomaly scorers over pretrained-CNN embeddings + classical CV features.
Caches features to features.npz so scoring is fast to re-run.

Usage:
  python solve.py extract     # extract + cache features for all images
  python solve.py score       # compute ensemble anomaly ranks -> ranked.csv + montages
  python solve.py all         # extract + score
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

# CPU gentleness: constrain threading before heavy libs load
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "images"
FEAT_CACHE = ROOT / "features.npz"
RANK_OUT = ROOT / "ranked.csv"
MONTAGE_DIR = ROOT / "montages"


def list_images():
    return sorted(DATA.glob("*.png"))


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_features(paths, cache_path=FEAT_CACHE):
    import cv2
    import torch
    from torchvision import models, transforms

    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    device = "cpu"
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Load pretrained models
    r18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
    r50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(device).eval()

    def emb(model, x):
        with torch.no_grad():
            # forward through conv+pool before fc
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            x = model.avgpool(x)
            return x.flatten(1)

    # incremental cache
    done = {}
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        done = {n: z[n] for n in z.files}

    names = [p.name for p in paths]
    names_all = names

    R18 = done.get("r18", np.zeros((0, 512), np.float32))
    R50 = done.get("r50", np.zeros((0, 2048), np.float32))
    CLS = done.get("cls", np.zeros((0, 0), np.float32))
    done_names = list(done.get("names", []))

    todo = [i for i, n in enumerate(names) if n not in done_names]
    print(f"[extract] {len(names)} total, {len(names) - len(todo)} cached, {len(todo)} to do", flush=True)

    if todo:
        r18_new = np.zeros((len(todo), 512), np.float32)
        r50_new = np.zeros((len(todo), 2048), np.float32)
        cls_new = np.zeros((len(todo), 0), np.float32)

        B = 32
        for s in range(0, len(todo), B):
            batch = []
            for i in todo[s:s + B]:
                img = cv2.imread(str(paths[i]))
                if img is None:
                    print("read fail", paths[i], flush=True)
                    batch.append(torch.zeros(3, 224, 224))
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    batch.append(tf(img))
            x = torch.stack(batch).to(device)
            e18 = emb(r18, x).cpu().numpy()
            e50 = emb(r50, x).cpu().numpy()
            r18_new[s:s + B] = e18
            r50_new[s:s + B] = e50
            if s % (B * 8) == 0:
                print(f"  {s}/{len(todo)}", flush=True)

        # classical features
        cls_new = compute_classical([paths[i] for i in todo])

        R18 = np.vstack([R18, r18_new]) if len(R18) else r18_new
        R50 = np.vstack([R50, r50_new]) if len(R50) else r50_new
        CLS = np.vstack([CLS, cls_new]) if len(CLS) else cls_new
        done_names = done_names + [names[i] for i in todo]

        np.savez_compressed(
            cache_path, names=np.array(done_names), r18=R18, r50=R50, cls=CLS
        )
        print(f"[extract] cached {len(done_names)} features -> {cache_path}", flush=True)

    return R18, R50, CLS


def compute_classical(paths):
    import cv2
    from skimage import feature as skf

    out = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            out.append(np.zeros(0, np.float32))
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        feats = []
        # HSV histograms
        for ch in range(3):
            h = cv2.calcHist([hsv], [ch], None, [16], [0, 256]).flatten()
            h = h / (h.sum() + 1e-9)
            feats.append(h.astype(np.float32))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # edges
        e = cv2.Canny(gray, 80, 200)
        feats.append(np.array([e.mean() / 255.0], np.float32))
        # LBP
        lbp = skf.local_binary_pattern(gray, P=8, R=1, method="uniform")
        lbp_hist = cv2.calcHist([lbp.astype('uint8')], [0], None, [10], [0, 10]).flatten()
        lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-9)
        feats.append(lbp_hist.astype(np.float32))
        # GLCM-ish stats
        g = gray.astype(np.float32)
        feats.append(np.array([g.mean() / 255.0, g.std() / 255.0], np.float32))
        # color moments
        for ch in range(3):
            c = img[:, :, ch].astype(np.float32)
            feats.append(np.array([c.mean(), c.std()], np.float32) / 255.0)
        out.append(np.concatenate(feats))
    n = max(len(f) for f in out)
    return np.vstack([np.pad(f, (0, n - len(f))) for f in out]).astype(np.float32)


# ---------------------------------------------------------------------------
# Anomaly scoring
# ---------------------------------------------------------------------------
def ml_scores(X):
    """Return dict of method -> normalized anomaly score (higher=more anomalous)."""
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
    from sklearn.decomposition import PCA

    scores = {}
    X = np.asarray(X, np.float32)

    # kNN distance
    k = min(10, len(X) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, _ = nn.kneighbors(X)
    scores["knn"] = dist[:, 1:].mean(axis=1)

    # IsolationForest (contamination ~2.3%)
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=0)
    iso.fit(X)
    scores["iso"] = -iso.score_samples(X)

    # LOF
    lof = LocalOutlierFactor(n_neighbors=min(20, len(X) - 1), contamination=0.05, novelty=False)
    scores["lof"] = -lof.fit_predict(X)  # placeholder replaced below
    scores["lof"] = -lof.negative_outlier_factor_

    # PCA reconstruction error (keep 90% variance)
    pca = PCA(n_components=0.9, random_state=0).fit(X)
    Xr = pca.inverse_transform(pca.transform(X))
    scores["pca"] = ((X - Xr) ** 2).sum(axis=1)

    # Mahalanobis-ish via robust center
    c = np.median(X, axis=0)
    scores["center"] = ((X - c) ** 2).sum(axis=1)

    return scores


def normalize_rank(s):
    order = np.argsort(np.argsort(s))
    return order / max(1, len(order) - 1)


def compute_ensemble(scores):
    keys = list(scores.keys())
    acc = np.zeros_like(scores[keys[0]], np.float64)
    for k in keys:
        acc += normalize_rank(scores[k])
    return acc / len(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["extract", "score", "all"], default="all")
    a = ap.parse_args()

    paths = list_images()
    print(f"[solve] {len(paths)} images", flush=True)

    if a.mode in ("extract", "all"):
        R18, R50, CLS = extract_features(paths)
    else:
        z = np.load(FEAT_CACHE, allow_pickle=True)
        R18, R50, CLS = z["r18"], z["r50"], z["cls"]
        names = list(z["names"])

    # standardize features
    def zs(X):
        m = X.mean(0, keepdims=True)
        s = X.std(0, keepdims=True) + 1e-8
        return (X - m) / s

    names = [p.name for p in paths]

    print("[solve] scoring r18 ...", flush=True)
    s18 = ml_scores(zs(R18))
    print("[solve] scoring r50 ...", flush=True)
    s50 = ml_scores(zs(R50))
    print("[solve] scoring classical ...", flush=True)
    scl = ml_scores(zs(CLS))

    # build combined score
    all_keys = set(s18) | set(s50) | set(scl)
    agg = np.zeros(len(names), np.float64)
    cnt = 0
    for k in all_keys:
        if k in s18:
            agg += normalize_rank(s18[k]); cnt += 1
        if k in s50:
            agg += normalize_rank(s50[k]); cnt += 1
        if k in scl:
            agg += normalize_rank(scl[k]); cnt += 1
    agg /= cnt
    rank = agg  # higher = more anomalous

    order = np.argsort(-rank)
    rows = [f"rank,name,score"]
    for i in order:
        rows.append(f"{i},{names[i]},{rank[i]:.6f}")
    RANK_OUT.write_text("\n".join(rows) + "\n")
    print(f"[solve] wrote {RANK_OUT}", flush=True)

    top60 = [names[i] for i in order[:60]]
    print("TOP 60:")
    for n in top60:
        print("  ", n)

    # build montages
    import cv2
    MONTAGE_DIR.mkdir(exist_ok=True)
    n = 60
    cells = [cv2.imread(str(DATA / top60[i])) for i in range(n)]
    cells = [c if c is not None else np.zeros((512, 512, 3), np.uint8) for c in cells]
    # 10 cols x 6 rows at 256 each
    tile = 256
    cols, rows_m = 10, 6
    canvas = np.zeros((rows_m * tile, cols * tile, 3), np.uint8)
    for i, c in enumerate(cells):
        r, cc = divmod(i, cols)
        csmall = cv2.resize(c, (tile, tile))
        canvas[r * tile:(r + 1) * tile, cc * tile:(cc + 1) * tile] = csmall
    # annotate
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(n):
        r, cc = divmod(i, cols)
        cv2.putText(canvas, top60[i], (cc * tile + 4, r * tile + 20), font, 0.45, (0, 255, 255), 1)
    cv2.imwrite(str(MONTAGE_DIR / "top60.png"), canvas)
    print(f"[solve] montage -> {MONTAGE_DIR / 'top60.png'}", flush=True)


if __name__ == "__main__":
    main()