#!/usr/bin/env python3
"""Vision-model scan helper for task_laos.

Uses NovAI glm-4.6v-flash (free, OpenAI-compatible) to visually inspect shell
images and return per-image "defective?" verdicts. Supports:
  - scan all / subset, either one image per call (full res) or montage grids.
  - saves incremental JSONL so runs can be resumed.

Usage examples:
  python vision_scan.py --mode full --names names_200.txt
  python vision_scan.py --mode montage --grid 4x4 --all
"""
from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np
import urllib.request

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "images"
API_KEY = "nvai-b5f01f382c5cbd6dd865cccbea63cec64014c1bbc7c98cde"
URL = "https://aiapi-pro.com/v1/chat/completions"
MODEL = "glm-4.6v-flash"


def chat_vision(messages, max_tokens=2000, retries=5):
    body = json.dumps({"model": MODEL, "messages": messages, "max_tokens": max_tokens})
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                URL, data=body.encode(),
                headers={"Content-Type": "application/json",
                         "Authorization": "Bearer " + API_KEY},
            )
            r = urllib.request.urlopen(req, timeout=240)
            d = json.loads(r.read())
            return d["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  [retry {attempt}] {e}", flush=True)
            time.sleep(3 + attempt * 3)
    return None


def img_b64(p):
    return base64.b64encode(open(str(p), "rb").read()).decode()


def build_montage(names, grid):
    cols, rows = grid
    tile = 512
    canvas = np.zeros((rows * tile, cols * tile, 3), np.uint8)
    for i, n in enumerate(names):
        if i >= cols * rows:
            break
        p = DATA / n
        c = cv2.imread(str(p))
        if c is None:
            c = np.zeros((512, 512, 3), np.uint8)
        r, cc = divmod(i, cols)
        canvas[r * tile:(r + 1) * tile, cc * tile:(cc + 1) * tile] = c
    # downscale to keep under size limit
    max_side = 1600
    h, w = canvas.shape[:2]
    if max(h, w) > max_side:
        sc = max_side / max(h, w)
        canvas = cv2.resize(canvas, (int(w * sc), int(h * sc)))
    ok, buf = cv2.imencode(".png", canvas)
    return base64.b64encode(buf.tobytes()).decode()


PROMPT_FULL = (
    "You are a quality inspector for old artillery/missile shells. "
    "This image shows one shell. Determine if the shell has any DEFECT that would "
    "indicate it might still be live/dangerous: a visible fuse/detonator/primer, a crack, "
    "a hole, a dent, a missing part, or an obvious anomaly. "
    "Most shells are normal (empty/safe). Answer with exactly one JSON line: "
    '{"defect": true/false, "reason": "<short reason>"}. Be very careful and precise.'
)

PROMpt_MONTAGE = (
    "This is a grid of {cols}x{rows} identical-size photos of old artillery/missile shells, "
    "numbered left-to-right, top-to-bottom starting at 1. Most shells are normal (empty/safe). "
    "A DEFECTIVE shell has a visible fuse/detonator/primer, a crack, a hole, a dent, a missing part, "
    "or an obvious anomaly. List the grid numbers (1-based) of any shells that look defective, "
    "with a one-line reason each. If none look defective, say NONE. "
    "Be precise; only flag clear defects."
)


def load_done(path):
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            done[d.get("image")] = d
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "montage"], default="full")
    ap.add_argument("--names", help="file with image names, one per line (else all images)")
    ap.add_argument("--grid", default="2x3")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "vision_results.jsonl"))
    a = ap.parse_args()

    if a.names:
        names = [l.strip() for l in Path(a.names).read_text().splitlines() if l.strip()]
    else:
        names = sorted(p.name for p in DATA.glob("*.png"))
    if a.limit:
        names = names[:a.limit]

    out_path = Path(a.out)
    done = load_done(out_path)

    if a.mode == "full":
        for idx, n in enumerate(names):
            if n in done:
                continue
            print(f"[{idx+1}/{len(names)}] {n}", flush=True)
            msg = [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img_b64(DATA / n)}},
                {"type": "text", "text": PROMPT_FULL},
            ]}]
            resp = chat_vision(msg)
            d = {"image": n, "raw": resp}
            with open(out_path, "a") as f:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    else:
        cols, rows = map(int, a.grid.split("x"))
        per = cols * rows
        for s in range(0, len(names), per):
            batch = names[s:s + per]
            if len(batch) == 0:
                break
            # skip fully-done batches
            if all(n in done for n in batch):
                continue
            print(f"[{s+1}/{len(names)}] montage of {len(batch)}", flush=True)
            b64 = build_montage(batch, (cols, rows))
            prompt = PROMpt_MONTAGE.format(cols=cols, rows=rows)
            msg = [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
                {"type": "text", "text": prompt},
            ]}]
            resp = chat_vision(msg)
            for i, n in enumerate(batch):
                if n in done:
                    continue
                d = {"image": n, "grid_pos": i + 1, "batch_start": s, "raw": resp}
                with open(out_path, "a") as f:
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print("DONE ->", out_path)


if __name__ == "__main__":
    main()