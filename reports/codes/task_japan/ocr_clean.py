#!/usr/bin/env python3
"""Clean OCR of caption bars. Median-filter grain removal + upscale + threshold."""
from __future__ import annotations
import glob, os, subprocess, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage

TESS = "/tmp/opencode/tessdata"
DATA = Path("challenges/task_japan/data/dataset")

def ocr_caption(path, psm=6):
    im = Image.open(path).convert("L")
    bar = im.crop((0, 755, 1280, 960))
    a = np.asarray(bar, dtype=np.float32)
    # median filter to kill grain, then normalize
    a = ndimage.median_filter(a, size=3)
    lo = a.min(); hi = a.max()
    a = np.clip((a - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype(np.uint8)
    # upscale
    a = np.asarray(Image.fromarray(a).resize((1280*3, 205*3), Image.LANCZOS))
    # threshold: background is dark, text is bright
    t = a.mean() + a.std()
    a = np.where(a > t, 255, 0).astype(np.uint8)
    Image.fromarray(a).save("/tmp/opencode/_ocr.png")
    r = subprocess.run(["tesseract", "/tmp/opencode/_ocr.png", "stdout", "-l", "jpn+eng", "--psm", str(psm)],
                       capture_output=True, text=True, env={**os.environ, "TESSDATA_PREFIX": TESS})
    return r.stdout.strip()

def main():
    args = sys.argv[1:]
    people = ["toorisugari", "yakin", "amayadori", "shuuden", "jihanki", "rojiura"]
    target = args[0] if args else "toorisugari"
    fs = []
    if target in people:
        fs = sorted(glob.glob(f"challenges/task_japan/data/dataset/credited/{target}/*.jpg"))
        out = f"challenges/task_japan/captions_{target}.json"
    elif target == "anonymous":
        fs = sorted(glob.glob("challenges/task_japan/data/dataset/anonymous/*.jpg"))
        out = "challenges/task_japan/captions_anonymous.json"
    else:
        sys.exit("usage: person | anonymous")
    results = {}
    for f in fs:
        name = os.path.basename(f)
        results[name] = ocr_caption(f)
        print(f"{name}: {results[name].splitlines()[:1]}", flush=True)
    with open(out, "w") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=1)
    print("saved", out, len(results))

if __name__ == "__main__":
    main()