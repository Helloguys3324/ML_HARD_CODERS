#!/usr/bin/env python
"""Classical sheep detector experiments (no training)."""
import os, glob
import numpy as np
import pandas as pd
import cv2
from skimage.feature import blob_log, peak_local_max
from skimage.morphology import white_tophat, disk, black_tophat

DATA = '/home/kozak/mllm/MML/work/tasks/task_mongolia/data'
OUT = '/home/kozak/mllm/MML/work/solutions/task_mongolia'


def sheep_score(img_rgb):
    """Return float score map: bright, low-saturation blobs."""
    img = img_rgb.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1] / 255.0, hsv[:, :, 2] / 255.0
    # whiteness: high value, low saturation
    white = v * (1.0 - s) ** 1.2
    # local background contrast via white top-hat at sheep scale
    w8 = (white * 255).astype(np.uint8)
    th = white_tophat(w8, disk(11)).astype(np.float32) / 255.0
    # dark sheep: black top-hat on V
    v8 = (v * 255).astype(np.uint8)
    bth = black_tophat(v8, disk(9)).astype(np.float32) / 255.0
    return white, th, bth


def detect(img_rgb, thr_ratio=0.25, min_sigma=2.5, max_sigma=8, overlap=0.3,
           use_dark=True, dark_ratio=0.5):
    white, th, bth = sheep_score(img_rgb)
    # normalize tophat by its high percentile
    scale = np.percentile(th, 99.5) + 1e-6
    score = th / scale
    blobs = blob_log(score, min_sigma=min_sigma, max_sigma=max_sigma,
                     num_sigma=6, threshold=thr_ratio, overlap=overlap)
    pts = [(b[1], b[0]) for b in blobs]
    if use_dark:
        dscale = np.percentile(bth, 99.5) + 1e-6
        dscore = bth / dscale
        dblobs = blob_log(dscore, min_sigma=min_sigma, max_sigma=max_sigma,
                          num_sigma=4, threshold=dark_ratio, overlap=overlap)
        pts += [(b[1], b[0]) for b in dblobs]
    return pts, score


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    counts = pd.read_csv(os.path.join(DATA, 'train_counts.csv'))
    ptsdf = pd.read_csv(os.path.join(DATA, 'train_points.csv'))
    sel = ['train_000.jpg', 'train_001.jpg', 'train_003.jpg', 'train_005.jpg',
           'train_010.jpg', 'train_020.jpg', 'train_002.jpg', 'train_004.jpg']
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    for ax, n in zip(axs.ravel(), sel):
        im = cv2.cvtColor(cv2.imread(os.path.join(DATA, 'train', n)), cv2.COLOR_BGR2RGB)
        pts, score = detect(im)
        true = int(counts.set_index('image').loc[n, 'count'])
        ax.imshow(im)
        if pts:
            p = np.array(pts); ax.scatter(p[:, 0], p[:, 1], s=14, facecolors='none', edgecolors='lime', linewidths=1)
        q = ptsdf[ptsdf.image == n]
        ax.scatter(q.x, q.y, s=6, c='red', marker='+', linewidths=0.8)
        ax.set_title(f'{n} true={true} det={len(pts)}'); ax.axis('off')
    plt.tight_layout(); plt.savefig(os.path.join(OUT, 'classical_detect.png'), dpi=80)
    print('saved')
