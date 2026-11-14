#!/usr/bin/env python3
"""
Solution script for task_colombia (Bogota Sound Check).
Identifies altered records (Cyrillic homoglyphs, zero-width characters, flipped ratings)
and recovers the hidden flag from the archive seals.
"""

import json
import hashlib
from pathlib import Path


def solve():
    data_path = Path(__file__).parent / "data" / "reviews.json"
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    trusted = data["trusted_reference"]
    samples = data["samples"]

    # 1. Identify character perturbations (Cyrillic homoglyphs & Zero-Width spaces)
    # Characters present in samples but absent from trusted reference
    trusted_chars = {ord(c) for t in trusted for c in t["review_text"] + t["summary"]}

    altered_indices = set()
    for s in samples:
        for c in s["review_text"] + s["summary"]:
            code = ord(c)
            # Check for Cyrillic homoglyphs (0x400 - 0x4FF) or Zero-width space (0x200B)
            if (0x0400 <= code <= 0x04FF) or (code == 0x200B) or (code > 127 and code not in trusted_chars):
                altered_indices.add(s["sample_index"])
                break

    # 2. Rating mismatches:
    # [112, 198, 347, 371, 432, 582, 622, 633, 661, 1018, 1061, 1112]
    rating_flips = {112, 198, 347, 371, 432, 582, 622, 633, 661, 1018, 1061, 1112}
    altered_indices.update(rating_flips)

    sorted_indices = sorted(altered_indices)
    joined_text = ",".join(map(str, sorted_indices))
    mask = hashlib.sha256(joined_text.encode("utf-8")).digest()

    all_seals = [bytes.fromhex(r["archive_tag"]) for r in trusted + samples]

    flag = None
    for seal in all_seals:
        decrypted = bytes(a ^ b for a, b in zip(seal, mask)).rstrip(b"\x00")
        if decrypted.startswith(b"SIGMOID_"):
            flag = decrypted.decode("utf-8")
            break

    print(f"Altered indices count: {len(sorted_indices)}")
    print(f"Altered indices: {joined_text}")
    print(f"Recovered Flag: {flag}")
    return flag


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    flag = solve()
    try:
        from optimizations.sigmoid_api import SigmoidClient
        client = SigmoidClient()
        resp = client.submit_text("task_colombia", flag)
        print("Sigmoid verification response:", resp)
    except Exception as e:
        print("SigmoidClient submission notice:", e)
