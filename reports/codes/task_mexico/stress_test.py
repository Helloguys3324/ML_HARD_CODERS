"""Stress-test main.solve() on harder synthetic corruption of the practice data.

The record ids stay the same and the ground truth is the practice output, so
we can measure exactly whether extra damage breaks the pipeline.
"""
import csv
import json
import os
import random
import re
import sys
import tempfile
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main as M

DATA = "/home/kozak/mllm/MML/work/tasks/task_mexico/data"
FIELDS = ["source_table", "record_id", "first_name", "last_name", "address"]

UNSEEN_ABBR = [("avenida", "avda"), ("interior", "intr"),
               ("departamento", "dept"), ("callejon", "callej"),
               ("boulevard", "bld"), ("prolongacion", "prolng"),
               ("circunvalacion", "circl"), ("edificio", "edif.")]

STREETS = ["Insurgentes", "Reforma", "Juarez", "Hidalgo", "Zaragoza",
           "Morelos", "Allende", "Madero", "Constitución", "Revolución",
           "Independencia", "Guerrero", "Matamoros", "Aldama", "Bravo"]


def nfkd(s):
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def corrupt(records, rng, p):
    """Apply random per-record damage; ground truth is unchanged."""
    out = []
    for r in records:
        r = dict(r)
        addr = r["address"]
        if rng.random() < p:
            # corrupt a zip digit
            addr = re.sub(r"\b(\d{5})\b",
                          lambda m: m.group(1)[:-1] + str(rng.randint(0, 9)),
                          addr, count=1)
        if rng.random() < p:
            # drop one token in the street line
            parts = addr.split(",")
            toks = parts[0].split()
            if len(toks) > 2:
                toks.pop(rng.randrange(len(toks) // 2))
                parts[0] = " ".join(toks)
                addr = ",".join(parts)
        if rng.random() < p:
            # unseen abbreviation form
            for full, short in UNSEEN_ABBR:
                if re.search(r"\b" + full + r"\b", addr, re.I):
                    addr = re.sub(r"\b" + full + r"\b", short, addr, count=1,
                                  flags=re.I)
                    break
        if rng.random() < p * 0.4:
            # replace the street name entirely (hard)
            parts = addr.split(",")
            toks = parts[0].split()
            idxs = [i for i, t in enumerate(toks)
                    if t.isalpha() and len(t) > 4]
            if idxs:
                toks[idxs[0]] = rng.choice(STREETS)
                parts[0] = " ".join(toks)
                addr = ",".join(parts)
        r["address"] = addr
        if rng.random() < p:
            for f in ("first_name", "last_name"):
                if r[f]:
                    v = list(r[f])
                    v[rng.randrange(len(v))] = rng.choice("abcdefghijklmnop")
                    r[f] = "".join(v)
        if rng.random() < p * 0.5:
            r["first_name"], r["last_name"] = r["last_name"], r["first_name"]
        if rng.random() < p * 0.3:
            r["first_name"] = (r["first_name"][:1] + ".") if r["first_name"] else "X."
        out.append(r)
    return out


def check(records, labels, tag):
    rng = random.Random(1234)
    recs = corrupt(records, rng, 0.08)
    rng.shuffle(recs)
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "test-input.csv")
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(recs)
        got = M.solve(p)
    A = set(frozenset(x) for x in got["matches"])
    B = set(frozenset(x) for x in labels)
    print(f"[{tag}] expected={len(B)} got={len(A)} missing={len(B-A)} extra={len(A-B)}")
    if B - A:
        print("   missing:", [tuple(x) for x in list(B - A)[:5]])
    if A - B:
        print("   extra  :", [tuple(x) for x in list(A - B)[:5]])
    return not (B - A or A - B)


if __name__ == "__main__":
    rows = list(csv.DictReader(open(os.path.join(DATA, "train-input.csv"),
                                    encoding="utf-8-sig")))
    labels = json.load(open(os.path.join(DATA, "train-output.json")))["matches"]
    ok = check(rows, labels, "hard-corruption")
    print("PASS" if ok else "FAIL")
