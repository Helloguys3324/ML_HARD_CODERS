#!/usr/bin/env python3
"""Parallel, resumable downloader for a subset of the task_china files.

Targets train/images only (the template-building dependency), using the cached
file list. Skipps already-downloaded files, retries with backoff.
"""
from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

LIST_CACHE = Path("/home/kozak/mllm/challenges/task_china/_file_list.json")
OUT = Path("/home/kozak/mllm/challenges/task_china/data")
PREFIX = "train/images/"


def fetch(file_id: str, dest: Path, session: requests.Session) -> bool:
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    res = session.get(url, timeout=180)
    if "download-form" in res.text:
        action = re.search(r'action="([^"]+)"', res.text).group(1)
        params = dict(re.findall(r'name="([^"]+)"\s+value="([^"]+)"', res.text))
        res = session.get(action, params=params, timeout=300)
    if res.status_code != 200 or len(res.content) < 100:
        return False
    head = res.content[:300]
    if head.lstrip().startswith(b"<!DOCTYPE") or b"<html" in head:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(res.content)
    tmp.replace(dest)
    return True


def download_one(item: dict) -> str | None:
    dest = OUT / item["path"]
    if dest.exists() and dest.stat().st_size > 500:
        return None  # already have it
    s = requests.Session()
    for attempt in range(8):
        try:
            if fetch(item["id"], dest, s):
                return None
        except Exception:
            pass
        time.sleep(1.5 * (attempt + 1))
    return item["path"]  # failed


def main() -> int:
    files = json.loads(LIST_CACHE.read_text())
    targets = [f for f in files if f["path"].startswith(PREFIX)]
    # prune already-downloaded
    todo = []
    have = 0
    for f in targets:
        d = OUT / f["path"]
        if d.exists() and d.stat().st_size > 500:
            have += 1
        else:
            todo.append(f)
    print(f"train images: {have} present, {len(todo)} to download", flush=True)

    failed = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(download_one, f): f["path"] for f in todo}
        done = 0
        for fut in as_completed(futs):
            done += 1
            err = fut.result()
            if err:
                failed.append(err)
            if done % 100 == 0:
                print(f"[i] {done}/{len(todo)} train, failed={len(failed)}", flush=True)
    print(f"[done] train images downloaded, failed={len(failed)}", flush=True)
    if failed:
        print("FAILED:\n" + "\n".join(failed[:50]), flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())