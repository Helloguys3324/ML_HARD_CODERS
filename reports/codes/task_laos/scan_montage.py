#!/usr/bin/env python3
"""Montage-based vision scan for task_laos (Bad BOMB).

Sends grids of shell images to a free vision LLM (NovAI glm-4.6v-flash) and
records which grid cells it flags as defective. Light on CPU (network-bound).
Resumable via batches.jsonl + results.jsonl.

Modes:
  scan   -- grid 3x3 over all images, lean-to-recall prompt
  verify -- full-res individual confirmation of a candidate list
  parse  -- summarize scans into a ranked defective list
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import urllib.request

cv2.setNumThreads(1)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "images"
API_KEY = "nvai-b5f01f382c5cbd6dd865cccbea63cec64014c1bbc7c98cde"
URL = "https://aiapi-pro.com/v1/chat/completions"
MODEL = "glm-4.6v-flash"


def chat_vision(messages, max_tokens=3000, retries=6):
    body = json.dumps({"model": MODEL, "messages": messages, "max_tokens": max_tokens})
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                URL, data=body.encode(),
                headers={"Content-Type": "application/json",
                         "Authorization": "Bearer " + API_KEY},
            )
            r = urllib.request.urlopen(req, timeout=300)
            d = json.loads(r.read())
            return d["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  [retry {attempt}] {e}", flush=True)
            time.sleep(2 + attempt * 4)
    return None


def img_b64(p):
    return base64.b64encode(Path(p).read_bytes()).decode()


def build_montage(names, cols, rows, tile=512, max_side=1600):
    canvas = np.zeros((rows * tile, cols * tile, 3), np.uint8)
    for i, n in enumerate(names):
        if i >= cols * rows:
            break
        c = cv2.imread(str(DATA / n))
        if c is None:
            c = np.zeros((512, 512, 3), np.uint8)
        r, cc = divmod(i, cols)
        canvas[r * tile:(r + 1) * tile, cc * tile:(cc + 1) * tile] = c
    h, w = canvas.shape[:2]
    if max(h, w) > max_side:
        sc = max_side / max(h, w)
        canvas = cv2.resize(canvas, (int(w * sc), int(h * sc)))
    ok, buf = cv2.imencode(".png", canvas)
    return base64.b64encode(buf.tobytes()).decode()


SCAN_PROMPT = (
    "This is a grid of {cols}x{rows} photos of old artillery/missile shells, numbered "
    "left-to-right, top-to-bottom starting at 1. Most shells are NORMAL (empty, safe, intact). "
    "A DEFECTIVE shell is one that shows ANY sign it might still be LIVE or damaged: "
    "a fuse/detonator/primer on the nose or base, a crack, a hole, a dent, a missing/loose part, "
    "rust/corrosion, or any visible anomaly on the shell body.\n"
    "Carefully inspect EVERY cell. For each defective cell, output one JSON object with "
    '"cell": <1-based number> and "reason": "<short>". Put them in a JSON array. '
    "If none are defective, output an empty array []. Output ONLY the JSON array."
)

VERIFY_PROMPT = (
    "Inspect this photo of an artillery/missile shell. Most shells are NORMAL (empty/safe/intact). "
    "Is this shell DEFECTIVE? Look for: a fuse/detonator/primer, a crack, a hole, a dent, a missing "
    "part, rust/corrosion, or any anomaly on the shell body. "
    'Answer with exactly one JSON object: {"defect": true/false, "confidence": 0-100, "reason": "<short>"}.'
)


def load_jsonl(path):
    rows = []
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def scan_all(cols, rows, out_batches, out_results):
    names = sorted(p.name for p in DATA.glob("*.png"))
    done = {r["batch_start"] for r in load_jsonl(out_batches)}
    per = cols * rows
    for s in range(0, len(names), per):
        if s in done:
            continue
        batch = names[s:s + per]
        print(f"[scan {s}/{len(names)}] {batch[0]}..{batch[-1]}", flush=True)
        b64 = build_montage(batch, cols, rows)
        prompt = SCAN_PROMPT.format(cols=cols, rows=rows)
        msg = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
            {"type": "text", "text": prompt},
        ]}]
        resp = chat_vision(msg)
        rec = {"batch_start": s, "names": batch, "raw": resp}
        with open(out_batches, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        # parse now
        items = parse_cells(resp, len(batch))
        for it in items:
            cell = it.get("cell")
            if isinstance(cell, int) and 1 <= cell <= len(batch):
                r = {"image": batch[cell - 1], "batch_start": s, "cell": cell,
                     "reason": it.get("reason", "")}
                with open(out_results, "a") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_cells(resp, n):
    """Extract JSON array of {cell, reason} from a model response, robustly."""
    if not resp:
        return []
    try:
        m = re.search(r"\[[\s\S]*\]", resp)
        if m:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                return [x for x in arr if isinstance(x, dict)]
    except Exception:
        pass
    # fallback: find "cell": N
    out = []
    for m in re.finditer(r'"cell"\s*:\s*(\d+)', resp):
        out.append({"cell": int(m.group(1))})
    # fallback: numbers like "1, 5, 9" after "defective/defect"
    if not out:
        seg = re.split(r"defect|anomal", resp, flags=re.I)
        if len(seg) > 1:
            for num in re.findall(r"\b\d{1,2}\b", resp):
                c = int(num)
                if 1 <= c <= n:
                    out.append({"cell": c})
    return out


def verify(names_file, out_verify):
    names = [l.strip() for l in Path(names_file).read_text().splitlines() if l.strip()]
    done = {r["image"] for r in load_jsonl(out_verify)}
    for i, n in enumerate(names):
        if n in done:
            continue
        print(f"[verify {i+1}/{len(names)}] {n}", flush=True)
        msg = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img_b64(DATA / n)}},
            {"type": "text", "text": VERIFY_PROMPT},
        ]}]
        resp = chat_vision(msg, max_tokens=500)
        with open(out_verify, "a") as f:
            f.write(json.dumps({"image": n, "raw": resp}, ensure_ascii=False) + "\n")


def parse_summary():
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["scan", "verify"])
    ap.add_argument("--grid", default="3x3")
    ap.add_argument("--names", default=str(ROOT / "candidates.txt"))
    ap.add_argument("--batches", default=str(ROOT / "batches.jsonl"))
    ap.add_argument("--results", default=str(ROOT / "scan_results.jsonl"))
    ap.add_argument("--verify-out", default=str(ROOT / "verify_results.jsonl"))
    a = ap.parse_args()

    if a.mode == "scan":
        cols, rows = map(int, a.grid.split("x"))
        scan_all(cols, rows, a.batches, a.results)
    elif a.mode == "verify":
        verify(a.names, a.verify_out)


if __name__ == "__main__":
    main()