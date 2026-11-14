#!/usr/bin/env python3
"""Solutions for "Ruta de las Identidades" (task_mexico).

Entity resolution over three merged registration tables.  Each row has
``source_table, record_id, first_name, last_name, address`` and the task is to
emit every unordered pair of record ids that belongs to the same person, as
``{"matches": [[id_a, id_b], ...]}`` (all pairs of a cluster, e.g. 3 pairs for
a cluster of 3).

How it is run
-------------
The grader copies this file into a scratch directory together with the hidden
``test-input.csv`` and runs ``python main.py`` (same convention as the other
LMML file tasks).  This script therefore:

* finds the input CSV from argv / environment / ``test-input.csv`` /
  ``train-input.csv`` / any CSV with the required header,
* writes ``test-output.json`` next to the working directory (and next to the
  input file), and also prints the same JSON object on stdout,
* exposes ``solve(csv_path) -> dict`` for import-based harnesses.

Method
------
The address is the stable identifier: after canonicalisation (accents,
punctuation, component re-ordering, and the register's abbreviation table such
as ``C.jón`` -> ``callejon``, ``Int.`` -> ``interior``) the records of one
person carry the *same multiset* of address tokens, while different people
essentially never do.  Names are heavily corrupted (typos, swaps, truncation)
and are used only as a secondary check / rescue signal.

Pipeline:
1. Normalise names and addresses (NFKD accent-strip, lowercase, punctuation,
   abbreviation expansion).
2. Candidate generation with several independent blocking keys (zip5, zip4,
   house number + street, rare name tokens, exact name) so a corrupted field
   cannot hide a true pair.
3. Score each candidate with token-set Jaccard, fuzzy token F1, character
   bigram Dice and character-level ratio on the canonical address, plus a
   fuzzy name score.
4. Threshold with a wide margin calibrated on the labelled practice data
   (all true pairs score 1.0 on address Jaccard; non-matches <= 0.42).
5. Union-find over accepted edges; a component may contain at most one record
   per source table (a safety conflict check), then every pair inside each
   component is emitted.

Standard library only, deterministic, ``O(#candidates * address length)``.
"""

from __future__ import annotations

import csv
import difflib
import glob
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache

# --------------------------------------------------------------------------
# Tunable, calibrated on the labelled practice set (see solution.md)
# --------------------------------------------------------------------------
ADDR_ACCEPT = 0.60        # canonical token Jaccard: accept immediately
RESCUE_ACCEPT = 0.80      # fuzzy/char combined score for the rescue branch
RESCUE_NAME = 0.55        # minimum fuzzy name score for the rescue branch
RESCUE_FLOOR = 0.20       # below this Jaccard the rescue metrics are pointless
SAME_SOURCE_ACCEPT = 0.90  # same-table pairs need this Jaccard ...
SAME_SOURCE_NAME = 0.60    # ... and this name score (generator: none exist)
MAX_BLOCK = 400           # skip pathological blocking buckets

REQUIRED = ("source_table", "record_id", "first_name", "last_name", "address")

# tolerant header aliases (the documented schema is the first entry of each)
FIELD_ALIASES = {
    "source_table": ("source_table", "source", "table", "src", "source_name"),
    "record_id": ("record_id", "recordid", "record", "id", "rid",
                  "record_key", "record_no"),
    "first_name": ("first_name", "firstname", "given_name", "givenname",
                   "nombres", "nombre"),
    "last_name": ("last_name", "lastname", "surname", "family_name",
                  "apellidos", "apellido"),
    "address": ("address", "addr", "full_address", "direccion", "domicilio"),
}

# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------
ADDR_MAP = {
    "av": "avenida", "avn": "avenida", "avenida": "avenida",
    "c": "calle", "cl": "calle", "calle": "calle",
    "cjon": "callejon", "callejon": "callejon", "jon": "callejon",
    "cor": "corredor", "corredor": "corredor",
    "peat": "peatonal", "peatonal": "peatonal",
    "priv": "privada", "privada": "privada",
    "prol": "prolongacion", "prolong": "prolongacion",
    "prolongacion": "prolongacion",
    "circ": "circunvalacion", "circunv": "circunvalacion",
    "circunvalacion": "circunvalacion",
    "calz": "calzada", "calzada": "calzada",
    "ret": "retorno", "retorno": "retorno",
    "ampl": "ampliacion", "ampli": "ampliacion", "ampliacion": "ampliacion",
    "edif": "edificio", "edificio": "edificio",
    "depto": "departamento", "dept": "departamento",
    "departamento": "departamento",
    "int": "interior", "interior": "interior",
    "and": "andador", "andador": "andador",
    "blvd": "boulevard", "boul": "boulevard", "boulevard": "boulevard",
    "no": "numero", "num": "numero", "numero": "numero",
    "pas": "pasaje", "pasaje": "pasaje",
    "cont": "continuacion", "continuacion": "continuacion",
    "perif": "periferico", "periferico": "periferico",
    "diag": "diagonal", "diagonal": "diagonal",
    "clz": "calzada",
}


def strip_accents(s):
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def norm(s):
    """Accent-free, lowercase, punctuation -> space, collapsed whitespace."""
    s = strip_accents(str(s if s is not None else ""))
    s = s.lower()
    s = re.sub(r"[^a-z0-9\u00f1]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def addr_tokens(address):
    """Canonical address tokens: abbreviations expanded, 'C.jón' merged."""
    raw = norm(address).split()
    out = []
    i = 0
    n = len(raw)
    while i < n:
        t = raw[i]
        if t == "c" and i + 1 < n and raw[i + 1] == "jon":
            out.append("callejon")
            i += 2
            continue
        if t == "q" and i + 1 < n and raw[i + 1] == "roo":
            out.append("qroo")
            i += 2
            continue
        out.append(ADDR_MAP.get(t, t))
        i += 1
    return out


def name_tokens(first, last):
    return [t for t in norm(str(first or "") + " " + str(last or "")).split() if t]


# --------------------------------------------------------------------------
# Similarities
# --------------------------------------------------------------------------
def jaccard(a, b):
    A, B = set(a), set(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def dice_bigrams(a, b):
    def bg(s):
        s = " " + s + " "
        return Counter(s[i:i + 2] for i in range(len(s) - 1))
    A, B = bg(a), bg(b)
    n = sum(A.values()) + sum(B.values())
    if not n:
        return 0.0
    return 2.0 * sum((A & B).values()) / n


@lru_cache(maxsize=500_000)
def _token_match(x, y):
    if x == y:
        return 1.0
    if len(x) == 1 or len(y) == 1:
        return 1.0 if x[0] == y[0] else 0.0
    if x.startswith(y) or y.startswith(x):
        return 0.95
    return difflib.SequenceMatcher(None, x, y).ratio()


def fuzzy_set_f1(A, B):
    """Symmetric best-match token similarity (handles unseen abbreviations).

    Tokens present in both sets contribute 1.0 directly; only the symmetric
    differences need the (comparatively expensive) fuzzy comparison.
    """
    if not A or not B:
        return 0.0
    sa, sb = set(A), set(B)
    common = len(sa & sb)
    ua = [x for x in sa if x not in sb]
    ub = [y for y in sb if y not in sa]
    fwd = sum(max(_token_match(x, y) for y in ub) for x in ua) if ub else 0.0
    bwd = sum(max(_token_match(y, x) for x in ua) for y in ub) if ua else 0.0
    return ((common + fwd) / len(sa) + (common + bwd) / len(sb)) / 2.0


# --------------------------------------------------------------------------
# Blocking
# --------------------------------------------------------------------------
def zip5(address):
    z = re.findall(r"\b\d{5}\b", address)
    return z[-1] if z else None


def house_number(tokens):
    for t in tokens:
        if t.isdigit() and 1 <= len(t) <= 4:
            return t
    return None


STREET_STOP = {
    "calle", "avenida", "callejon", "corredor", "peatonal", "privada",
    "prolongacion", "circunvalacion", "calzada", "retorno", "ampliacion",
    "edificio", "departamento", "interior", "andador", "boulevard", "pasaje",
    "continuacion", "periferico", "diagonal", "vial", "eje", "numero",
    "norte", "sur", "oriente", "poniente", "de", "del", "la", "los", "las",
    "y", "san", "santa", "nueva", "nuevo", "vieja", "viejo",
}


def street_sig(tokens):
    for t in tokens:
        if not t.isdigit() and t not in STREET_STOP and len(t) >= 3:
            return t
    return None


def candidate_pairs(records, addr_cache, name_cache):
    """Union of several blocking keys -> sorted unique index pairs."""
    blocks = defaultdict(list)

    def add(key, idx):
        blocks[key].append(idx)

    # document frequency of name tokens, to use only rare ones as blockers
    df = Counter()
    for nm in name_cache:
        df.update(set(nm))

    for idx, r in enumerate(records):
        addr = r["address"]
        toks = addr_cache[idx]
        z = zip5(addr)
        if z:
            add(("z5", z), idx)
            add(("z4", z[:4]), idx)
        hn = house_number(toks)
        ss = street_sig(toks)
        if hn and ss:
            add(("hs", hn, ss[:6]), idx)
        nm = tuple(sorted(name_cache[idx]))
        if nm:
            add(("nm", nm), idx)
        for t in set(name_cache[idx]):
            if len(t) >= 5 and df[t] <= 25:
                add(("nt", t), idx)

    pairs = set()
    for key, idxs in blocks.items():
        if len(idxs) < 2 or len(idxs) > MAX_BLOCK:
            continue
        v = sorted(set(idxs))
        for a in range(len(v)):
            ia = v[a]
            for b in range(a + 1, len(v)):
                pairs.add((ia, v[b]))
    return pairs


# --------------------------------------------------------------------------
# Scoring / matching
# --------------------------------------------------------------------------
class Matcher:
    def __init__(self, records):
        self.records = records
        self.addr_tokens = [addr_tokens(r["address"]) for r in records]
        self.name_tokens = [name_tokens(r["first_name"], r["last_name"])
                            for r in records]
        self.addr_sets = [set(t) for t in self.addr_tokens]
        self.addr_str = [" ".join(t) for t in self.addr_tokens]
        self.num_sets = [{t for t in toks if t.isdigit()}
                         for toks in self.addr_tokens]

    def name_score(self, i, j):
        return fuzzy_set_f1(self.name_tokens[i], self.name_tokens[j])

    def anchored(self, i, j):
        """True when the two records share a stable address anchor.

        The register preserves digits and the house number/street name; only
        abbreviations/order change.  Requiring an anchor keeps the fuzzy
        rescue branch from firing on accidental word overlap (country names,
        state codes, ...).
        """
        za = zip5(self.records[i]["address"])
        zb = zip5(self.records[j]["address"])
        if za and za == zb:
            return True
        A, B = self.addr_tokens[i], self.addr_tokens[j]
        ha, hb = house_number(A), house_number(B)
        if ha and ha == hb:
            sa, sb = street_sig(A), street_sig(B)
            if sa and sb and (sa == sb or sa.startswith(sb[:4])
                              or sb.startswith(sa[:4])):
                return True
        # The register preserves numeric components (house / apartment / zip
        # extension); two shared numbers is a strong identity anchor even when
        # a street name or the main zip was damaged.
        if len(self.num_sets[i] & self.num_sets[j]) >= 2:
            return True
        return False

    def evaluate(self, i, j, want_name=False):
        """Return (jaccard, combined_address_score, name_score_or_None).

        The expensive fuzzy/character metrics only run for pairs whose cheap
        token Jaccard is inside the rescue band: the practice data separates
        perfectly (true pairs 1.0, non-matches <= 0.42), so this is purely a
        hedge against unseen abbreviation/corruption variants.
        """
        jac = jaccard(self.addr_sets[i], self.addr_sets[j])
        if jac >= ADDR_ACCEPT:
            return jac, jac, (self.name_score(i, j) if want_name else None)
        if jac < RESCUE_FLOOR:
            return jac, jac, (self.name_score(i, j) if want_name else None)
        fuzzy = fuzzy_set_f1(self.addr_tokens[i], self.addr_tokens[j])
        big = dice_bigrams(self.addr_str[i], self.addr_str[j])
        seq = difflib.SequenceMatcher(None, self.addr_str[i],
                                      self.addr_str[j]).ratio()
        combined = max(jac, 0.97 * fuzzy, 0.95 * big, 0.95 * seq)
        return jac, combined, self.name_score(i, j)


def build_components(records, edges):
    """Union-find with a 'one record per source table' safety constraint.

    ``edges`` is a list of (score, i, j, same_source).  Highest score first.
    A merge is skipped when it would put two records of the same source table
    into one component (the practice generator guarantees one record per
    person and table; this kills residual false positives).
    """
    n = len(records)
    parent = list(range(n))
    sources = [{records[i]["source_table"]} for i in range(n)]
    members = {i: [i] for i in range(n)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for score, i, j, same in sorted(edges, key=lambda e: -e[0]):
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        if not same and (sources[ri] & sources[rj]):
            continue
        if ri > rj:
            ri, rj = rj, ri
        parent[rj] = ri
        sources[ri] |= sources[rj]
        members[ri].extend(members.pop(rj))
    return [members[x] for x in members]


def _norm_col(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _map_columns(fieldnames):
    """Map normalised CSV headers to the logical schema fields."""
    by_norm = {}
    for name in fieldnames or []:
        by_norm.setdefault(_norm_col(name), name)
    mapping = {}
    for want, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            if alias in by_norm:
                mapping[want] = by_norm[alias]
                break
    # a single 'name' column instead of first/last names
    if "first_name" not in mapping and "last_name" not in mapping:
        for cand in ("name", "full_name", "fullname", "nombre_completo"):
            if cand in by_norm:
                mapping["name"] = by_norm[cand]
                break
    return mapping


def _read_head(path):
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                return f.readline()
        except UnicodeDecodeError:
            continue
    return ""


def _looks_like_header(head, delim):
    cols = [_norm_col(c) for c in head.split(delim)]
    has_id = any(c in ("id", "record", "recordid", "rid", "record_key")
                 or c.endswith("_id") for c in cols)
    has_addr = any("address" in c or c in ("addr", "direccion", "domicilio")
                   for c in cols)
    return has_id and has_addr


def _detect_delimiter(path):
    head = _read_head(path)
    for delim in (",", ";", "\t", "|"):
        if _looks_like_header(head, delim):
            return delim
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            dialect = csv.Sniffer().sniff(f.read(8192), delimiters=",;\t|")
        if dialect.delimiter:
            return dialect.delimiter
    except (OSError, csv.Error):
        pass
    return ","


def solve(csv_path):
    delimiter = _detect_delimiter(csv_path)
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        mapping = _map_columns(reader.fieldnames)
        rows = []
        for raw in reader:
            rec = {}
            for want in REQUIRED:
                col = mapping.get(want)
                rec[want] = (raw.get(col) or "").strip() if col else ""
            if "name" in mapping and not (rec["first_name"] or rec["last_name"]):
                parts = str(raw.get(mapping["name"]) or "").strip().split(None, 1)
                rec["first_name"] = parts[0] if parts else ""
                rec["last_name"] = parts[1] if len(parts) > 1 else ""
            if rec["record_id"]:
                rows.append(rec)
    if not rows:
        return {"matches": []}

    # de-duplicate identical record ids deterministically
    seen = set()
    records = []
    for r in rows:
        if r["record_id"] in seen:
            continue
        seen.add(r["record_id"])
        if not r["source_table"]:
            # defensive: derive a table id when the column is missing/empty
            mprefix = re.match(r"^[A-Za-z]+", r["record_id"])
            r["source_table"] = (mprefix.group(0).lower() if mprefix
                                 else "_row%d" % len(records))
        records.append(r)

    m = Matcher(records)
    cands = candidate_pairs(records, m.addr_tokens, m.name_tokens)

    edges = []
    for i, j in cands:
        same = records[i]["source_table"] == records[j]["source_table"]
        jac, combined, ns = m.evaluate(i, j, want_name=same)
        if same:
            if ns is None:
                ns = m.name_score(i, j)
            if jac >= SAME_SOURCE_ACCEPT and ns >= SAME_SOURCE_NAME:
                edges.append((jac, i, j, True))
        else:
            accept = jac >= ADDR_ACCEPT or (
                jac >= RESCUE_FLOOR and combined >= RESCUE_ACCEPT
                and ns is not None and ns >= RESCUE_NAME and m.anchored(i, j))
            if accept:
                edges.append((combined if jac < ADDR_ACCEPT else jac, i, j, False))

    comps = build_components(records, edges)
    out = []
    for comp in comps:
        if len(comp) < 2:
            continue
        comp = sorted(comp, key=lambda k: records[k]["record_id"])
        for a in range(len(comp)):
            for b in range(a + 1, len(comp)):
                out.append([records[comp[a]]["record_id"],
                            records[comp[b]]["record_id"]])
    out.sort()
    return {"matches": out}


# --------------------------------------------------------------------------
# I/O plumbing (grader conventions: test-input.csv -> test-output.json)
# --------------------------------------------------------------------------
def _has_header(path):
    try:
        head = _read_head(path)
        return bool(head) and _looks_like_header(head, _detect_delimiter(path))
    except OSError:
        return False


def find_input():
    here = os.path.dirname(os.path.abspath(__file__))
    for a in sys.argv[1:]:
        if a and os.path.isfile(a) and _has_header(a):
            return a
    for var in ("INPUT_FILE", "INPUT_PATH", "INPUT_CSV", "DATA_FILE"):
        p = os.environ.get(var)
        if p and os.path.isfile(p) and _has_header(p):
            return p
    names = ("test-input.csv", "train-input.csv", "input.csv", "data.csv",
             "input-data.csv", "test.csv")
    for base in (os.getcwd(), here):
        for name in names:
            p = os.path.join(base, name)
            if os.path.isfile(p) and _has_header(p):
                return p
    for base in (os.getcwd(), here):
        for pattern in ("*.csv", "*.CSV"):
            for p in sorted(glob.glob(os.path.join(base, pattern))):
                if "output" not in os.path.basename(p).lower() and _has_header(p):
                    return p
    raise FileNotFoundError(
        "no input CSV with columns source_table,record_id,first_name,last_name,address"
    )


def main():
    try:
        in_path = find_input()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    result = solve(in_path)
    payload = json.dumps(result, ensure_ascii=False)
    # file output (platform convention: test-output.json)
    targets = {os.path.join(os.getcwd(), "test-output.json")}
    targets.add(os.path.join(os.path.dirname(os.path.abspath(in_path)),
                             "test-output.json"))
    for t in targets:
        try:
            with open(t, "w", encoding="utf-8") as f:
                f.write(payload)
        except OSError:
            pass
    print(payload)
    print(f"[task_mexico] {len(result['matches'])} pairs from {in_path}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
