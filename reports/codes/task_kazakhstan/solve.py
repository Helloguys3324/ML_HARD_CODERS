#!/usr/bin/env python3
"""Stamp of Disapproval (task_kazakhstan) -- forensic workbench.

Reproduces the analysis described in solution.md:

  1. Segment the three stamps on every 1600x1200 album page.
  2. Characterise the three alteration families shown by the 30 labelled
     exhibits (mask PNGs):
        - "flat fill"      : high-pass energy collapses to ~0 inside a
                             rectangle; often an exactly constant RGB colour;
        - "ghost removal"  : text survives as a low-contrast ghost inside a
                             rectangle (alpha-blended toward paper colour);
        - "digit/element"  : the pasted patch is sharper / has a different
                             JPEG recompression signature than its host.
  3. Score local rectangular anomalies with several cues (high-pass
     flatness, exact-colour rectangularity, ELA at the *native* JPEG
     quality, ELA/texture residual) and write per-album dot plots
     (one dot per page) -- the geometry that is supposed to spell a letter.

Usage:
    python solve.py --data dataset [--out outdir] [--album N]

Requires only numpy + opencv-python (same versions as requirements.txt:
numpy==2.2.6, opencv-python==5.0.0.93).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

PAGE_W, PAGE_H = 1600, 1200
BG = np.array([38, 36, 34])          # stock-book background colour
MQ = 97                              # measured native JPEG quality (quant tables 1,1,1,1,1,2,3,4)


# --------------------------------------------------------------------------- #
# segmentation
# --------------------------------------------------------------------------- #
def stamps_on(page: np.ndarray):
    """Connected components of everything that is not the dark album board."""
    d = np.abs(page.astype(np.int16) - BG).max(2)
    m = cv2.morphologyEx((d > 30).astype(np.uint8), cv2.MORPH_CLOSE,
                         np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    boxes = []
    for x, y, w, h, a in stats[1:]:
        if w > 100 and h > 100:
            boxes.append((int(x), int(y), int(w), int(h)))
    # order top-to-bottom, left-to-right (stable page reading order)
    boxes.sort(key=lambda b: (b[1] // 200, b[0]))
    return boxes


# --------------------------------------------------------------------------- #
# forensic maps
# --------------------------------------------------------------------------- #
def highpass(gray: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    g = gray.astype(np.float32)
    return np.abs(g - cv2.GaussianBlur(g, (0, 0), sigma))


def local_std(gray: np.ndarray, k: int = 7) -> np.ndarray:
    g = gray.astype(np.float32)
    m = cv2.boxFilter(g, -1, (k, k))
    m2 = cv2.boxFilter(g * g, -1, (k, k))
    return np.sqrt(np.maximum(0, m2 - m * m))


def ela(page: np.ndarray, q: int = MQ) -> np.ndarray:
    """Error-level analysis at the page's native JPEG quality."""
    ok, enc = cv2.imencode(".jpg", page, [cv2.IMWRITE_JPEG_QUALITY, q])
    again = cv2.imdecode(enc, 1)
    return np.abs(page.astype(np.float32) - again.astype(np.float32)).mean(2)


def exact_flat_components(im: np.ndarray, min_area: int = 400):
    """Rectangles that survived JPEG as an *exactly* constant RGB colour.

    A painted fill in a single colour has zero AC coefficients, so at q~97 it
    decodes back to exactly equal bytes.  Scanned paper never does that over a
    large area.  Returns [(score, x, y, w, h, area, colour_step), ...].
    """
    key = ((im[:, :, 0].astype(np.int32) << 16)
           | (im[:, :, 1].astype(np.int32) << 8)
           | im[:, :, 2].astype(np.int32))
    eq = np.zeros(key.shape, bool)
    eq[:, :-1] |= key[:, :-1] == key[:, 1:]
    eq[:-1, :] |= key[:-1, :] == key[1:, :]
    m = cv2.morphologyEx(eq.astype(np.uint8), cv2.MORPH_OPEN,
                         np.ones((7, 7), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    H, W = im.shape[:2]
    out = []
    for i, (x, y, w, h, a) in enumerate(stats[1:], 1):
        if a < min_area or w < 25 or h < 18 or a / (w * h) < 0.75:
            continue
        x0, y0 = max(0, x - 12), max(0, y - 12)
        x1, y1 = min(W, x + w + 12), min(H, y + h + 12)
        inside = (lab[y:y + h, x:x + w] == i)
        ring = np.ones((y1 - y0, x1 - x0), bool)
        ring[y - y0:y + h - y0, x - x0:x + w - x0] = False
        ring &= (lab[y0:y1, x0:x1] == 0)
        if ring.sum() < 100:
            continue
        ci = im[y:y + h, x:x + w][inside].reshape(-1, 3).mean(0)
        co = im[y0:y1, x0:x1][ring].reshape(-1, 3).mean(0)
        step = float(np.abs(ci - co).mean())
        out.append((float(step * min(1.0, w * h / 1500.0)),
                    x, y, w, h, a, step))
    out.sort(reverse=True)
    return out


def ela_residual_map(page: np.ndarray):
    """ELA after removing the part explained by local texture slope.

    Edited patches have either a much lower ELA (flat fills) or a much higher
    one (freshly rendered glyphs) than natural content of the same texture.
    """
    e = ela(page)
    g = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 4.0
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 4.0
    t = np.sqrt(gx * gx + gy * gy)
    m = t < 50
    if m.sum() < 1000:
        m = np.ones_like(t, bool)
    a = np.polyfit(t[m].ravel(), e[m].ravel(), 1)
    return e - (a[0] * t + a[1]), t


# --------------------------------------------------------------------------- #
# exhibit calibration
# --------------------------------------------------------------------------- #
def exhibit_report(data: Path):
    """Summarise what an altered patch looks like inside each mask."""
    rows = []
    for mp in sorted((data / "exhibits").glob("*_mask.png")):
        jpg = mp.with_name(mp.name.replace("_mask.png", ".jpg"))
        im = cv2.imread(str(jpg))
        mk = cv2.imread(str(mp), 0)
        x, y, w, h = cv2.boundingRect((mk > 127).astype(np.uint8))
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        hp = highpass(g)
        inside = hp[y:y + h, x:x + w]
        rest = hp.copy()
        rest[y:y + h, x:x + w] = np.nan
        rest = rest[g > 40]
        rows.append(dict(exhibit=int(mp.stem.split("_")[1]),
                         x=x, y=y, w=w, h=h,
                         hp_in=float(inside.mean()),
                         hp_out=float(np.nanmean(rest)),
                         ratio=float(inside.mean() / (np.nanmean(rest) + 1e-6)),
                         std_in=float(im[y:y + h, x:x + w].std()),
                         corner=f"{'T' if y < im.shape[0] / 2 else 'B'}"
                                f"{'L' if x < im.shape[1] / 2 else 'R'}"))
    return rows


# --------------------------------------------------------------------------- #
# album dot plots
# --------------------------------------------------------------------------- #
def album_dots(data: Path, album: int, outdir: Path, method: str):
    """Return one (x, y, detail) per page and draw the tracing-paper sheet."""
    pages = sorted((data / "albums" / f"album_{album}").glob("*.jpg"))
    sheet = np.zeros((PAGE_H, PAGE_W, 3), np.uint8)
    dots, boxes = [], []
    for f in pages:
        page = cv2.imread(str(f))
        sb = stamps_on(page)
        res, tmap = ela_residual_map(page)
        best = None
        for i, (x, y, w, h) in enumerate(sb):
            x0, y0, x1, y1 = x + 10, y + 10, x + w - 10, y + h - 10
            if x1 <= x0 or y1 <= y0:
                continue
            if method == "ela_residual":
                R = cv2.boxFilter(res[y0:y1, x0:x1], -1, (9, 9))
                my, mx = np.unravel_index(int(np.argmax(R)), R.shape)
                score = float(np.percentile(R, 95))
                cand = (score, x0 + int(mx), y0 + int(my), w, h)
            elif method == "flatfill":
                cand = None
                st = page[y:y + h, x:x + w]
                for c in exact_flat_components(st):
                    if cand is None or c[0] > cand[0]:
                        cand = (c[0], x + c[1] + c[3] // 2,
                                y + c[2] + c[4] // 2, w, h)
                if cand is None:
                    continue
            else:
                raise SystemExit("unknown method")
            if best is None or cand[0] > best[0]:
                best = cand
        if best:
            _, cx, cy, w, h = best
            dots.append((cx, cy))
            boxes.append((f.stem, cx - 25, cy - 25, 50, 50))
            cv2.circle(sheet, (cx, cy), 8, (0, 255, 255), -1)
    cv2.imwrite(str(outdir / f"dots_album{album}_{method}.png"),
                cv2.resize(sheet, (800, 600)))
    return dots, boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="out")
    ap.add_argument("--album", type=int, default=1)
    args = ap.parse_args()
    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rep = exhibit_report(data)
    with open(out / "exhibit_stats.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rep[0].keys()))
        wr.writeheader()
        wr.writerows(rep)
    print(f"{len(rep)} exhibits analysed -> {out/'exhibit_stats.csv'}")

    all_boxes = []
    for a in [args.album] if args.album else range(1, 6):
        for method in ("ela_residual", "flatfill"):
            dots, boxes = album_dots(data, a, out, method)
            print(f"album {a} {method}: {len(dots)} dots")
            for b in boxes:
                all_boxes.append([a, *b])
    with open(out / "boxes_candidates.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["album", "page", "x", "y", "w", "h"])
        wr.writerows(all_boxes)


if __name__ == "__main__":
    main()
