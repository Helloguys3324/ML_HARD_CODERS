#!/usr/bin/env python3
"""OCR the subtitle bar (y=768..960) of every credited photo; dump text + first chars."""
from __future__ import annotations
import glob, json, subprocess, os
from pathlib import Path
from PIL import Image
import numpy as np

TESS = os.environ.get("TESSDATA_PREFIX", "/tmp/opencode/tessdata")
DATA = Path("challenges/task_japan/data/dataset")

def ocr_bar(path, lang="jpn+eng"):
    im = Image.open(path).convert("L")
    bar = im.crop((0, 760, 1280, 960))
    bar = bar.resize((2560, 400), Image.LANCZOS)
    a = np.asarray(bar)
    a = np.clip((a - a.min()) / (a.max() - a.min() + 1e-9) * 255, 0, 255).astype(np.uint8)
    tmp = "/tmp/opencode/ocr_tmp.png"
    Image.fromarray(a).save(tmp)
    r = subprocess.run(["tesseract", tmp, "stdout", "-l", lang, "--psm", "6"],
                       capture_output=True, text=True, env={**os.environ, "TESSDATA_PREFIX": TESS})
    return r.stdout.strip()

def main():
    out = {}
    for p in sorted(DATA.glob("credited/*")):
        handle = p.name
        files = sorted(p.glob("*.jpg"))
        for f in files:
            txt = ocr_bar(f)
            out[f"{handle}/{f.name}"] = txt
            print(f"{handle:12s} {f.name} :: {txt.splitlines()[:1]}", flush=True)
    json.dump(out, open("challenges/task_japan/captions.json", "w"), ensure_ascii=False, indent=1)
    print("done", len(out))

if __name__ == "__main__":
    main()