#!/usr/bin/env python3
"""
task_japan — "City of Subtitles" (Jimaku no Machi)
Sensor fingerprint (PRNU) attribution of anonymous photos to 6 photographers,
then detect toorisugari's set and extract the hidden word.

Approach:
  1. Inventory.
  2. PRNU reference per photographer (mean wavelet noise residual over credited photos).
  3. Correlate each anonymous residual with each reference -> attribution.
  4. Secret extraction on toorisugari's subset.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.restoration import denoise_wavelet

DATA = Path(__file__).resolve().parent / "data" / "dataset"
CREDITED = DATA / "credited"
ANON = DATA / "anonymous"

PEOPLE = ["toorisugari", "yakin", "amayadori", "shuuden", "jihanki", "rojiura"]


def load_gray(path: Path) -> np.ndarray:
    im = Image.open(path).convert("L")
    return np.asarray(im, dtype=np.float64) / 255.0


def resize_to(A: np.ndarray, shape) -> np.ndarray:
    if A.shape == shape:
        return A
    im = Image.fromarray((np.clip(A, 0, 1) * 255).astype(np.uint8))
    return np.asarray(im.resize((shape[1], shape[0]), Image.BILINEAR), dtype=np.float64) / 255.0


def noise_residual(gray: np.ndarray) -> np.ndarray:
    d = denoise_wavelet(gray, method="BayesShrink", mode="soft",
                        wavelet_levels=4, rescale_sigma=True)
    return gray.astype(np.float64) - d.astype(np.float64)


def whiten(x: np.ndarray) -> np.ndarray:
    # remove mean and low-frequency trends via row/col mean subtraction
    x = x - x.mean()
    x = x - x.mean(axis=1, keepdims=True)
    x = x - x.mean(axis=0, keepdims=True)
    return x


def pce(corr: np.ndarray) -> float:
    """Peak-to-correlation-energy on the 2D correlation surface."""
    peak = corr.max()
    energy = (corr ** 2).sum() - peak ** 2
    if energy <= 0:
        return float(peak) if peak > 0 else 0.0
    return float(peak ** 2 / (energy / corr.size))


def correlate(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation (PCE) between two same-size residuals."""
    aw = whiten(a.astype(np.float64))
    bw = whiten(b.astype(np.float64))
    na = np.linalg.norm(aw)
    nb = np.linalg.norm(bw)
    if na == 0 or nb == 0:
        return 0.0
    # full 2D correlation via FFT, then PCE
    Fa = np.fft.rfft2(aw)
    Fb = np.fft.rfft2(bw)
    corr = np.fft.irfft2(Fa * np.conj(Fb), s=aw.shape)
    return pce(corr) / (na * nb)  # PCE of the NCC


def inventory() -> dict:
    out = {"people": {}, "anonymous": []}
    for p in PEOPLE:
        d = CREDITED / p
        files = sorted(d.glob("*.jpg")) if d.exists() else []
        hs = {}
        for f in files:
            im = Image.open(f)
            hs[f.name] = (im.size, im.mode)
        out["people"][p] = {"count": len(files), "sizes": hs}
    anon_files = sorted(ANON.glob("*.jpg"))
    out["anonymous"] = [f.name for f in anon_files]
    out["anonymous_count"] = len(anon_files)
    return out


def build_references():
    refs = {}
    for p in PEOPLE:
        d = CREDITED / p
        files = sorted(d.glob("*.jpg"))
        acc = None
        n = 0
        for f in files:
            g = load_gray(f)
            r = noise_residual(g)
            if acc is None:
                acc = r.copy()
            else:
                acc += r
            n += 1
        ref = acc / n
        refs[p] = whiten(ref)
    return refs


def attribute(refs: dict) -> dict:
    results = {}
    for fname in sorted(ANON.glob("*.jpg")):
        g = load_gray(fname)
        r = noise_residual(g)
        r = whiten(r)
        scores = {}
        for p, ref in refs.items():
            scores[p] = correlate(r, ref)
        best = max(scores, key=scores.get)
        results[fname.name] = {"best": best, "scores": scores}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", action="store_true")
    ap.add_argument("--attribute", action="store_true")
    ap.add_argument("--json-out", type=str, default=str(Path(__file__).parent / "attribution.json"))
    args = ap.parse_args()

    if args.inventory:
        inv = inventory()
        print(json.dumps(inv, indent=2, default=str))
        return

    if args.attribute:
        print("Building references...", file=sys.stderr)
        refs = build_references()
        print("Attributing...", file=sys.stderr)
        res = attribute(refs)
        # summarize
        from collections import Counter
        cnt = Counter(v["best"] for v in res.values())
        print(json.dumps({"counts_by_person": dict(cnt), "detail": res}, indent=2))
        with open(args.json_out, "w") as f:
            json.dump(res, f, indent=2)
        return

    ap.print_help()


if __name__ == "__main__":
    main()