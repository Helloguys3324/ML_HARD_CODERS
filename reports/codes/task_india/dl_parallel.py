#!/usr/bin/env python3
"""Parallel Google Drive folder downloader for task_india."""
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

FOLDER_URL = sys.argv[1] if len(sys.argv) > 1 else "https://drive.google.com/drive/folders/1i9HpSq7yYWfLxN_7kEbSlqY66aYPiF6M?usp=sharing"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("challenges/task_india/data")
import gdown

OUT.mkdir(parents=True, exist_ok=True)

files = gdown.download_folder(url=FOLDER_URL, skip_download=True, use_cookies=False) or []
print(f"[i] {len(files)} files listed", flush=True)

def fetch(f):
    dest = OUT / f.path
    if dest.exists() and dest.stat().st_size > 0:
        return (f.path, "skip", dest.stat().st_size)
    session = requests.Session()
    url = f"https://drive.google.com/uc?export=download&id={f.id}"
    try:
        res = session.get(url, timeout=60)
        if "download-form" in res.text:
            action = re.search(r'action="([^"]+)"', res.text).group(1)
            params = dict(re.findall(r'name="([^"]+)"\s+value="([^"]+)"', res.text))
            res = session.get(action, params=params, timeout=120)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(res.content)
        return (f.path, "ok", len(res.content))
    except Exception as e:
        return (f.path, "err", str(e))

done = 0
with ThreadPoolExecutor(max_workers=16) as ex:
    futs = {ex.submit(fetch, f): f for f in files}
    for fut in as_completed(futs):
        path, status, info = fut.result()
        done += 1
        if done % 200 == 0:
            print(f"[{done}/{len(files)}] {path} {status} {info}", flush=True)

print("DONE", flush=True)