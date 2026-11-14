#!/usr/bin/env python3
"""Cloudflare (Mistral Small 3.1 vision) individual shell inspector for task_laos.

Reliable full-resolution per-image defect verdicts. Network-bound, light CPU.
Resumable: appends to a JSONL of {image, defect, confidence, reason}.

Usage:
  python cf_scan.py all                  # every image
  python cf_scan.py file names.txt       # subset
  python cf_scan.py top N score.csv      # first N rows by rank (col2 = name)
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

import urllib.request

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "images"
ACCT = "41a6c273486f4c5850e48d3e9875a65a"
TOK = os.environ.get("CF_API_TOKEN", "")
MODEL = "@cf/mistralai/mistral-small-3.1-24b-instruct"
OUT = ROOT / "cf_results.jsonl"

PROMPT = (
    "You are a bomb-disposal inspector. Inspect this photo of an old artillery/missile shell. "
    "Most shells are NORMAL: empty, safe, intact metal. A DEFECTIVE shell shows at least one sign "
    "it might still be live or is damaged: a fuse/detonator/primer on the nose or base, a visible "
    "crack, a hole, a dent, a missing piece, heavy rust/corrosion, or any unusual feature on the "
    "shell body. Lighting and background vary and are NOT defects — judge only the shell itself.\n"
    "Reply with exactly one JSON object, no markdown: "
    '{"defect": true or false, "confidence": 0-100, "reason": "short"}'
)


def chat(img_b64, retries=5):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img_b64}},
        ]}],
        "max_tokens": 200,
    })
    url = f"https://api.cloudflare.com/client/v4/accounts/{ACCT}/ai/v1/chat/completions"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=body.encode(),
                                         headers={"Authorization": "Bearer " + TOK,
                                                  "Content-Type": "application/json"})
            r = urllib.request.urlopen(req, timeout=120)
            d = json.loads(r.read())
            content = d["choices"][0]["message"]["content"]
            neurons = d.get("usage", {}).get("neurons", 0)
            return content, neurons
        except urllib.error.HTTPError as e:
            err = e.read().decode()[:300]
            print(f"  [retry {attempt}] HTTP {e.code}: {err[:120]}", flush=True)
            if e.code == 429 or "neurons" in err.lower() or "quota" in err.lower() or "exhausted" in err.lower():
                time.sleep(30 + attempt * 30)
            else:
                time.sleep(3 + attempt * 3)
        except Exception as e:
            print(f"  [retry {attempt}] {e}", flush=True)
            time.sleep(3 + attempt * 3)
    return None, 0


def parse(content):
    m = re.search(r"\{[\s\S]*\}", content or "")
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    # fallback heuristics
    if content and re.search(r"\btrue\b", content, re.I) and re.search(r"defect", content, re.I):
        return {"defect": True, "confidence": 60, "reason": content[:80]}
    return {"defect": False, "confidence": 60, "reason": content[:80] if content else ""}


def load_done():
    done = {}
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                done[d["image"]] = d
            except Exception:
                pass
    return done


def resolve_names(args):
    if args.mode == "all":
        return sorted(p.name for p in DATA.glob("*.png"))
    if args.mode == "file":
        return [l.strip() for l in Path(args.arg).read_text().splitlines() if l.strip()]
    if args.mode == "top":
        n = args.count
        rows = []
        for line in Path(args.arg).read_text().splitlines()[1:]:  # skip header
            parts = line.split(",")
            if len(parts) >= 2:
                rows.append(parts[1])
        return rows[:n]
    raise SystemExit("bad mode")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["all", "file", "top"])
    ap.add_argument("arg", nargs="?", default=None)
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out_path = Path(a.out)

    names = resolve_names(a)
    done = load_done()
    todo = [n for n in names if n not in done]
    print(f"[cf_scan] {len(names)} names, {len(done)} done, {len(todo)} todo", flush=True)

    total_neurons = 0
    for i, n in enumerate(todo):
        p = DATA / n
        if not p.exists():
            continue
        b64 = base64.b64encode(p.read_bytes()).decode()
        content, neurons = chat(b64)
        total_neurons += neurons
        if content is None:
            print(f"[{i+1}/{len(todo)}] {n} FAILED", flush=True)
            continue
        res = parse(content)
        rec = {"image": n, **res, "raw": content, "neurons": neurons}
        with open(out_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{i+1}/{len(todo)}] {n} -> defect={res.get('defect')} conf={res.get('confidence')} "
              f"| neurons cum={total_neurons:.0f}", flush=True)
        time.sleep(0.4)

    print(f"DONE, total neurons ~{total_neurons:.0f} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()