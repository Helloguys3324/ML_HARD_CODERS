#!/usr/bin/env python3
"""
Ruta de las Identidades (task_mexico) - LMML 2026
Entity resolution / record linkage across 3 tables (a, b, c).
Identifies all matching pairs of records referring to the same person.
"""

import csv
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    from difflib import SequenceMatcher


def strip_accents(s: str) -> str:
    """Remove diacritics and return lowercase string."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    ).lower()


ABBREVIATIONS = {
    "av": "avenida",
    "c": "calle",
    "cjon": "callejon",
    "and": "andador",
    "priv": "privada",
    "ret": "retorno",
    "peat": "peatonal",
    "cor": "corredor",
    "blvd": "boulevard",
    "prol": "prolongacion",
    "ampl": "ampliacion",
    "edif": "edificio",
    "depto": "departamento",
    "int": "interior",
}


def normalize_address(addr: str) -> str:
    """Normalize address by standardizing street/unit abbreviations and sorting tokens."""
    s = strip_accents(addr)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    tokens = [ABBREVIATIONS.get(t, t) for t in s.split()]
    return " ".join(sorted(tokens))


def normalize_name(fn: str, ln: str) -> str:
    """Normalize name by stripping accents and sorting tokens (handles first/last transpositions)."""
    s = strip_accents(f"{fn} {ln}")
    tokens = re.findall(r"[a-z0-9]+", s)
    return " ".join(sorted(tokens))


def get_initials(fn: str, ln: str) -> set:
    """Extract set of initials from name tokens."""
    s = strip_accents(f"{fn} {ln}")
    tokens = re.findall(r"[a-z0-9]+", s)
    return {t[0] for t in tokens if t}


def compute_similarity(s1: str, s2: str) -> float:
    """Compute string similarity (0.0 to 1.0) using rapidfuzz or difflib fallback."""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    if HAS_RAPIDFUZZ:
        return fuzz.ratio(s1, s2) / 100.0
    return SequenceMatcher(None, s1, s2).ratio()


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[rx] = ry


def find_input_file() -> str:
    """Locate input CSV file."""
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        return sys.argv[1]

    candidates = [
        "test-input.csv",
        "train-input.csv",
        "data/test-input.csv",
        "data/train-input.csv",
        "challenges/task_mexico/data/test-input.csv",
        "challenges/task_mexico/data/train-input.csv",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("Could not find test-input.csv or train-input.csv")


def solve(input_csv: str) -> list:
    """Read CSV, perform blocked fuzzy matching and transitive closure."""
    with open(input_csv, mode="r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    # Preprocess records and build geographic blocks (State + 5-digit ZIP)
    postal_regex = re.compile(r"([A-Z\.\s]+)\s+(\d{5})(?:-\d{4})?$")
    blocks = defaultdict(list)

    for r in records:
        r["norm_addr"] = normalize_address(r["address"])
        r["norm_name"] = normalize_name(r["first_name"], r["last_name"])
        r["initials"] = get_initials(r["first_name"], r["last_name"])

        m = postal_regex.search(r["address"].strip())
        if m:
            key = (m.group(1).strip().upper(), m.group(2))
        else:
            d = re.findall(r"\b\d{5}\b", r["address"])
            key = ("UNKNOWN", d[0]) if d else ("UNKNOWN", "UNKNOWN")
        blocks[key].append(r)

    uf = UnionFind()

    # Pairwise comparison within each block
    for key, rows in blocks.items():
        n = len(rows)
        if n < 2:
            continue
        for i in range(n):
            r1 = rows[i]
            for j in range(i + 1, n):
                r2 = rows[j]
                # Only compare across distinct source tables
                if r1["source_table"] == r2["source_table"]:
                    continue

                # Address similarity check
                addr_sim = compute_similarity(r1["norm_addr"], r2["norm_addr"])
                if addr_sim < 0.75:
                    continue

                # Name similarity / initial compatibility check
                name_sim = compute_similarity(r1["norm_name"], r2["norm_name"])
                if name_sim >= 0.20 or (r1["initials"] & r2["initials"]):
                    uf.union(r1["record_id"], r2["record_id"])

    # Cluster components and emit all cross-table pairs (cliques)
    comps = defaultdict(list)
    for r in records:
        root = uf.find(r["record_id"])
        comps[root].append(r)

    matches = []
    seen = set()
    for root, cluster in comps.items():
        k = len(cluster)
        if k < 2:
            continue
        for i in range(k):
            for j in range(i + 1, k):
                if cluster[i]["source_table"] != cluster[j]["source_table"]:
                    pair = tuple(sorted([cluster[i]["record_id"], cluster[j]["record_id"]]))
                    if pair not in seen:
                        seen.add(pair)
                        matches.append([pair[0], pair[1]])

    return matches


def main():
    input_path = find_input_file()
    print(f"Reading input from: {input_path}")
    matches = solve(input_path)
    print(f"Found {len(matches)} matching pairs.")

    output_path = "test-output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"matches": matches}, f, indent=2)
    print(f"Wrote output to {output_path}")


if __name__ == "__main__":
    main()
