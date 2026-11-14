#!/usr/bin/env python3
"""
task_colombia — "Looks Good to Me! — Bogotá Sound Check"

Reproducible analysis of colombia_flag_data.zip / reviews.json.

What this script does
---------------------
1. Extracts reviews.json into ./data (never touches the original zip).
2. Scans `trusted_reference` + `samples` for character-level tampering
   (invisible / homoglyph characters) and shows the records that contain them.
3. Runs near-duplicate detection between `samples` and `trusted_reference`
   (character n-gram TF-IDF + difflib) and prints the strongest pairs.
4. Implements the archive_note procedure:
       mask = sha256( ",".join(sorted(sample_index)) )
       plaintext = archive_tag_bytes XOR mask, rstrip(b"\x00")
   and tests the candidate altered-record sets (invisible-char records,
   near-duplicate records, and their union) against every archive_tag.
5. Prints diagnostics: how many tags decode to printable text, longest
   trailing-zero run, etc.

The script intentionally does NOT submit anything.  Run with:
    /home/kozak/mllm/py work/solutions/task_colombia/solve.py
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ZIP = Path("/home/kozak/mllm/MML/work/tasks/task_colombia/data/colombia_flag_data.zip")
DATA = HERE / "data"
JSON_PATH = DATA / "reviews.json"

# Candidate phrase guess documented in solution.md (LOW confidence).
ANSWER_CANDIDATE = "SIGMOID_SOUNDCHECK"


def extract() -> None:
    if JSON_PATH.exists():
        return
    import zipfile

    DATA.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(DATA)


def load() -> tuple[list[dict], list[dict], str]:
    with open(JSON_PATH, encoding="utf-8") as f:
        d = json.load(f)
    return d["trusted_reference"], d["samples"], d["archive_note"]


def unicode_anomalies(trusted: list[dict], samples: list[dict]) -> list[int]:
    """Characters that occur in samples but never in the trusted reference.

    In this dataset the trusted reviews contain only two non-ASCII characters:
    U+2019 RIGHT SINGLE QUOTATION MARK and U+2013 EN DASH.  Samples contain a
    handful of extra characters: U+200B ZERO WIDTH SPACE and four Cyrillic
    homoglyphs (а е о с).  Records whose text/summary contains one of these
    are the visible character-level tampering.
    """
    trusted_chars: set[str] = set()
    for r in trusted:
        trusted_chars.update(r["review_text"])
        trusted_chars.update(r["summary"])
    allowed = trusted_chars

    print("\n[1] Characters present in samples but absent from trusted_reference:")
    weird = Counter()
    for r in samples:
        for f in ("review_text", "summary"):
            for ch in r[f]:
                if ch not in allowed:
                    weird[ch] += 1
    for ch, n in weird.most_common():
        print(f"    U+{ord(ch):04X} {unicodedata.name(ch, '?'):40s} x{n}")

    hit = []
    for r in samples:
        bad = [
            ch
            for f in ("review_text", "summary")
            for ch in r[f]
            if ch not in allowed
        ]
        if bad:
            hit.append(r["sample_index"])
            print(
                f"    S{r['sample_index']:4d}  chars={[f'U+{ord(c):04X}' for c in bad]}"
                f"  summary={r['summary'][:60]!r}"
            )
    return sorted(hit)


def near_duplicates(
    trusted: list[dict], samples: list[dict], top: int = 20
) -> list[tuple[float, int, int, str, str]]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    A = [r["review_text"].lower() for r in trusted]
    B = [r["review_text"].lower() for r in samples]
    v = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
        max_features=200_000,
    )
    Ca = v.fit_transform(A)
    Cb = v.transform(B)
    S = (Cb @ Ca.T).toarray()
    best = S.argmax(1)
    vals = S.max(1)

    order = np.argsort(-vals)[:top]
    out = []
    print("\n[2] Strongest sample -> trusted near-duplicates (char 3-5 gram cosine):")
    for j in order:
        i = int(best[j])
        out.append((float(vals[j]), int(samples[j]["sample_index"]), i,
                    samples[j]["summary"], trusted[i]["summary"]))
        print(
            f"    S{samples[j]['sample_index']:4d} ~ T{i:4d}  sim={vals[j]:.3f}"
            f"  ratings={samples[j]['rating']}/{trusted[i]['rating']}"
            f"  | {samples[j]['summary'][:38]!r} || {trusted[i]['summary'][:38]!r}"
        )
    return out


def tag_array(trusted: list[dict], samples: list[dict]) -> tuple[np.ndarray, list[tuple[str, int]]]:
    keys = [("T", i) for i in range(len(trusted))] + [
        ("S", r["sample_index"]) for r in samples
    ]
    tags = [bytes.fromhex(r["archive_tag"]) for r in trusted] + [
        bytes.fromhex(r["archive_tag"]) for r in samples
    ]
    return np.array([list(t) for t in tags], dtype=np.uint8), keys


def try_mask(
    tarr: np.ndarray,
    keys: list[tuple[str, int]],
    indices: list[int],
    label: str,
) -> None:
    txt = ",".join(map(str, sorted(indices)))
    mask = hashlib.sha256(txt.encode()).digest()
    B = tarr ^ np.frombuffer(mask, dtype=np.uint8)
    rev = B[:, ::-1]
    zero_run = (rev != 0).argmax(1)
    print(
        f"\n    [{label}] n={len(indices)} mask={mask.hex()[:16]}... "
        f"max_trailing_zero_run={int(zero_run.max())} "
        f"tags_with_run>=2={int((zero_run >= 2).sum())}"
    )
    hits = []
    for idx in range(len(B)):
        row = bytes(B[idx]).rstrip(b"\x00")
        if len(row) >= 6 and all(32 <= c < 127 for c in row):
            hits.append((keys[idx], row.decode("latin-1")))
    if hits:
        print(f"    printable decodes: {hits[:5]}")
    else:
        print("    no fully printable decode (message not recovered)")


def main() -> None:
    extract()
    trusted, samples, note = load()

    print("=" * 78)
    print("task_colombia — archive analysis")
    print("=" * 78)
    print(f"ZIP: {ZIP}")
    print(f"trusted_reference: {len(trusted)}   samples: {len(samples)}")
    print(f"archive_note: {note!r}")

    # 0. structure sanity
    assert all(r["sample_index"] == i for i, r in enumerate(samples))

    unicode_ids = unicode_anomalies(trusted, samples)

    near = near_duplicates(trusted, samples, top=20)
    near_ids = sorted({s for _, s, _, _, _ in near[:12]})
    five_ids = [593, 546, 606, 700, 936]  # strongest same-product edit pairs
    print(f"\n[3] Candidate altered sets")
    print(f"    invisible/homoglyph records: {unicode_ids}")
    print(f"    strongest near-duplicate records: {five_ids}")

    tarr, keys = tag_array(trusted, samples)

    print("\n[4] Testing the archive_note mask procedure on candidate sets")
    for label, ids in [
        ("invisible/homoglyph 12", unicode_ids),
        ("near-duplicate 5", five_ids),
        ("union 17", sorted(set(unicode_ids) | set(five_ids))),
    ]:
        try_mask(tarr, keys, ids, label)

    # 5. exhaustive subset search over the union of the 17 candidates
    union17 = sorted(set(unicode_ids) | set(five_ids))
    print(f"\n[5] Exhaustive subset search over union17 ({len(union17)} records, "
          f"{2**len(union17):,} subsets)")
    sig = b"SIGMOID_"
    allowed_prefix = {
        bytes(a ^ b for a, b in zip(bytes(row[:8]), sig)) for row in tarr
    }
    suff4 = {bytes(row[-4:]) for row in tarr}
    hits = 0
    for r in range(2, len(union17) + 1):
        for comb in itertools.combinations(union17, r):
            m = hashlib.sha256(",".join(map(str, comb)).encode()).digest()
            if m[:8] in allowed_prefix:
                print(f"    PREFIX HIT {comb} {m.hex()}")
                hits += 1
            if m[-4:] in suff4:
                B = tarr ^ np.frombuffer(m, dtype=np.uint8)
                ok = np.all((B[:, :8] >= 32) & (B[:, :8] < 127), axis=1)
                for idx in np.where(ok)[0]:
                    row = bytes(B[idx]).rstrip(b"\x00")
                    if len(row) >= 6 and all(32 <= c < 127 for c in row):
                        print(f"    DECODE HIT {comb} {keys[idx]} {row!r}")
                        hits += 1
    print(f"    subset hits: {hits}")

    # 6. summary / caveat
    print("\n[6] Conclusion")
    print(
        "    The mask procedure does not decode to readable text for the\n"
        "    visible tampering set (invisible/homoglyph records), for the strong\n"
        "    near-duplicate pairs, or for any subset of their union.\n"
        "    Therefore the exact altered-record set used by the task generator\n"
        "    is not recovered by these detectors (or the note's encoding differs).\n"
        "    Best-effort flag candidate written to answer.txt is a thematic guess."
    )

    # 7. write answer candidate
    (HERE / "answer.txt").write_text(ANSWER_CANDIDATE + "\n", encoding="utf-8")
    print(f"\n    wrote {HERE / 'answer.txt'} = {ANSWER_CANDIDATE}")


if __name__ == "__main__":
    sys.exit(main())
