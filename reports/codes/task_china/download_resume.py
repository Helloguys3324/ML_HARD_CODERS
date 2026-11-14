#!/usr/bin/env python3
"""Resilient Google Drive folder downloader with retries + resume.

Usage: .venv/bin/python challenges/task_china/download_resume.py
"""
from __future__ import annotations

import json
import re
import sys
import time
import traceback
from pathlib import Path

import gdown
import requests

URL = "https://drive.google.com/drive/folders/1TE_-nwZzBUTDztXivSbNqeyCFFaN1WDa?usp=sharing"
OUT = Path("/home/kozak/mllm/challenges/task_china/data")
LIST_CACHE = Path("/home/kozak/mllm/challenges/task_china/_file_list.json")


def get_file_list() -> list[dict]:
    if LIST_CACHE.exists():
        return json.loads(LIST_CACHE.read_text())
    files = gdown.download_folder(url=URL, skip_download=True, use_cookies=False) or []
    data = [{"id": f.id, "path": f.path} for f in files]
    LIST_CACHE.write_text(json.dumps(data))
    print(f"[i] listed {len(data)} files -> {LIST_CACHE.name}", flush=True)
    return data


def fetch(session: requests.Session, file_id: str, dest: Path) -> bool:
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    res = session.get(url, timeout=180, allow_redirects=True)
    if "download-form" in res.text:
        action = re.search(r'action="([^"]+)"', res.text).group(1)
        params = dict(re.findall(r'name="([^"]+)"\s+value="([^"]+)"', res.text))
        res = session.get(action, params=params, timeout=300)
    if res.status_code != 200 or len(res.content) < 100:
        return False
    # guard: reject google html error pages
    if res.content[:200].lstrip().startswith(b"<!DOCTYPE") or b"<html" in res.content[:200]:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(res.content)
    return True


def main() -> int:
    files = get_file_list()
    session = requests.Session()
    done = 0
    failed = []
    for i, f in enumerate(files):
        dest = OUT / f["path"]
        if dest.exists() and dest.stat().st_size > 0:
            done += 1
            continue
        ok = False
        for attempt in range(6):
            try:
                ok = fetch(session, f["id"], dest)
                if ok:
                    break
            except Exception as e:
                print(f"[w] {f['path']} attempt {attempt}: {type(e).__name__}", flush=True)
            time.sleep(2 * (attempt + 1))
        if ok:
            done += 1
        else:
            failed.append(f["path"])
        if (i + 1) % 50 == 0:
            print(f"[i] progress {i+1}/{len(files)} done={done} failed={len(failed)}", flush=True)
    print(f"[done] {done}/{len(files)} files, failed={len(failed)}", flush=True)
    if failed:
        print("FAILED:\n" + "\n".join(failed), flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())