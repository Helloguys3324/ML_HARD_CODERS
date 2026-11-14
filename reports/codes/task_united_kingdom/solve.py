"""
Call My BLEUff (task_united_kingdom) - solver.

Idea
----
The house scorer (bleu.py) is corpus-level BLEU-4 with:
  * no smoothing (any zero 4-gram precision => BLEU 0),
  * clipping against the single reference with the highest count,
  * brevity penalty from the *closest* reference length.

BLEU is precision-oriented: it only asks "how much of what you wrote occurs
in the references", never "did you say everything that is true".  The
panellists exploit this in three observable ways:

  1. COPY      : verbatim (or nearly verbatim) reference sentences, often
                 several references concatenated -> very high p1..p4.
  2. PAD       : concatenating two reference descriptions of the same card
                 repeats every fact, inflating matched n-gram counts.
  3. TRUNCATE  : cutting a reference phrase short ("... with a low.") raises
                 precision and keeps the brevity penalty mild, but is not a
                 good description.

Counter-metric used here (a listener's check, not BLEU's):

  badness(deck) = sum over its 40 cards of
      1.0 * (p4 >= 0.8)              # perfect 4-gram precision = copying
    + 1.0 * (sentence ~ reference)   # a sentence copied/near-copied
    + 1.0 * (cov5 >= 0.85)           # almost every 5-gram occurs in a ref
    + 0.7 * (length ratio extreme)   # truncation / padding
    + 1.0 * (fact recall < 0.55)     # most of the venue facts missing
    + 0.7 * (broken sentence)        # ends on a function word / word salad
    + 1.5 * (fact contradiction)     # contradicts the reference consensus
    + 0.5 * (padded length)          # >1.3x the closest reference length
    + 0.5 * (>=4 facts missing)
    + 0.7 * (duplicated fact mentions)

The honest panellist is the deck with the lowest badness.  Per-round the
minimum is unique in most rounds; the raw decoded word is reported for
transparency.  Finally we decode under the constraint that the 8-letter
sequence is a real word (candidate list from the system SCOWL en_GB
dictionary), choosing the word whose implied honest seats have the lowest
total badness.

Run:  /home/kozak/mllm/py solve.py
"""
from __future__ import annotations

import difflib
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "tasks" / "task_united_kingdom" / "data"
if not DATA.exists():
    DATA = Path(__file__).resolve().parents[2] / "tasks" / "task_united_kingdom" / "data"

TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tok(t: str) -> list[str]:
    return TOKEN.findall(t.lower())


def ngrams(tokens: list[str], n: int) -> set[tuple]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


# ---------------------------------------------------------------------------
# fact extraction (price / rating / family / area / cuisine)
# ---------------------------------------------------------------------------
PRICE = {
    "low": r"cheap|inexpensive|low[- ]priced|low price|low cost|low-cost|affordable|discount|cost-friendly|budget|less than £?20|under £?20|below £?20|less than 20 pounds",
    "med": r"moderate|mid[- ]range|medium[- ]priced|medium price|average[- ]priced|average price|£20-?25|20-25|around £?20|about £?20|reasonable|middle price",
    "high": r"expensive|high[- ]priced|high price|pricey|costly|higher price|high end|high-end|more than £?30|over £?30|£?30 and above|above £?30|at least £?30",
}
RATING = {
    "low": r"1 out of 5|one out of 5|one star|1 star|1-star|one-star|low customer rating|low rating|low rated|rated low|low reviews|bad rating|poorly rated|1/5",
    "med": r"3 out of 5|three out of 5|three star|3 star|3-star|three-star|average customer rating|average rating|averagely rated|average rated|moderate rating|moderately rated|average reviews|3/5",
    "high": r"5 out of 5|five out of 5|five star|5 star|5-star|five-star|high customer rating|high rating|highly rated|excellent rating|great rating|top rated|5/5",
}
AREA = {
    "riverside": r"riverside|by the river|on the river|riverfront|beside a river|near the river|along the river|north of the city centre",
    "city": r"city centre|city center|downtown|center of the city|centre of the city",
    "outskirts": r"outskirt",
}
CUISINE = {
    "japanese": r"japanese|sushi|oriental",
    "italian": r"italian",
    "french": r"french",
    "english": r"english|british",
    "indian": r"indian|curry",
    "chinese": r"chinese|dumpling",
    "fast": r"fast food|burger|fries",
}


def extract(text: str) -> dict:
    t = " " + re.sub(r"\s+", " ", text.lower()) + " "
    out = {}
    for group, name in ((PRICE, "price"), (RATING, "rating"), (AREA, "area"), (CUISINE, "cuisine")):
        found = {v for v, p in group.items() if re.search(p, t)}
        if found:
            out[name] = found.pop() if len(found) == 1 else "mixed"
    t2 = (t.replace("isn't", "is not").replace("aren't", "are not")
            .replace("isnt", "is not").replace("doesn't", "does not"))
    if re.search(r"adults only|adult only|adult oriented|no family amenities|no children policy|no kid", t2):
        out["family"] = "no"
    elif re.search(r"not\s+(?:very\s+)?(?:family|child|kid)[- ]?friendly|not for (?:the whole )?family|not family|leave the kids|non[- ]?family|non[- ]?kid", t2):
        out["family"] = "no"
    elif re.search(r"family[- ]friendly|kid[- ]?friendly|child[- ]?friendly|children[- ]?friendly|families|family oriented|welcome children|for the whole family", t2):
        out["family"] = "yes"
    return out


def consensus(refs: list[str]) -> dict:
    fs = [extract(r) for r in refs]
    truth = {}
    for k in ("price", "rating", "family", "area", "cuisine"):
        c = Counter(f.get(k) for f in fs if f.get(k) not in (None, "mixed"))
        if c:
            val, n = c.most_common(1)[0]
            if n >= 4 or (n == 3 and len(c) == 1):
                truth[k] = val
    return truth


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
def sentence_copied(text: str, refs: list[str]) -> int:
    ts = [tok(s) for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    rss = [[tok(s) for s in re.split(r"(?<=[.!?])\s+", r.strip()) if s.strip()] for r in refs]
    for t in ts:
        for ss in rss:
            for s in ss:
                if difflib.SequenceMatcher(None, t, s, autojunk=False).ratio() >= 0.95:
                    return 1
    return 0


def broken(text: str) -> int:
    toks = tok(text)
    if len(toks) >= 2 and toks[-1] in {"the", "a", "an", "of", "to", "for", "and", "or", "with", "by", "near", "at", "in", "on", "from", "than", "more"}:
        return 1
    low = " ".join(toks)
    for p in (r"\bis a an\b", r"\bwith is\b", r"\bof with\b", r"\bto and\b",
              r"\bmore than more than\b", r"\bnear to and\b", r"\bthe of\b",
              r"\bof of\b", r"\bthe the\b", r"\ba a\b"):
        if re.search(p, low):
            return 1
    return 0


def main() -> None:
    refs = pd.read_csv(DATA / "references.csv")
    panel = pd.read_csv(DATA / "panel.csv")
    ref_map = refs.groupby(["round", "card"]).reference.apply(list).to_dict()

    rows = []
    for _, row in panel.iterrows():
        rs = ref_map[(row["round"], row.card)]
        toks = tok(row.text)
        rts = [tok(r) for r in rs]
        cov5 = 0.0
        if len(toks) >= 5:
            t5 = ngrams(toks, 5)
            allref = set()
            for r in rts:
                allref |= ngrams(r, 5)
            cov5 = len(t5 & allref) / len(t5)
        # p4 against this card's references (house metric, single segment)
        import sys
        sys.path.insert(0, str(DATA))
        from bleu import segment_stats  # noqa: E402
        st = segment_stats(row.text, rs)
        p4 = st["matched"][3] / st["total"][3] if st["total"][3] else 0.0
        ratio = st["out_len"] / max(1, st["ref_len"])
        truth = consensus(rs)
        f = extract(row.text)
        wrong = sum(1 for k, v in truth.items() if f.get(k) not in (None, "mixed", v))
        nattr = len(truth)
        miss = sum(1 for k, v in truth.items() if f.get(k) is None)
        # duplicate fact mentions
        dup = 0
        for k, group in (("price", PRICE), ("rating", RATING), ("area", AREA)):
            c = sum(len(re.findall(p, " " + row.text.lower() + " ")) for p in group.values())
            if c > 1:
                dup += c - 1
        rows.append(dict(
            round=row["round"], card=row.card, seat=row.seat,
            cov5=cov5, p4=p4, ratio=ratio,
            exact=sentence_copied(row.text, rs), broken=broken(row.text),
            wrong=wrong, miss=miss, dup=dup, nattr=nattr,
            text=row.text,
        ))
    F = pd.DataFrame(rows)
    bad = (
        1.0 * (F.p4 >= 0.8)
        + 1.0 * F.exact
        + 1.0 * (F.cov5 >= 0.85)
        + 0.7 * (F.ratio < 0.72)
        + 1.0 * (F.miss / F.nattr.replace(0, 1) >= 0.45)
        + 0.7 * F.broken
        + 1.5 * (F.wrong > 0)
        + 0.5 * (F.ratio > 1.3)
        + 0.5 * (F.miss >= 4)
        + 0.7 * (F.dup >= 2)
    )
    F["bad"] = bad
    deck = F.groupby(["round", "seat"]).bad.sum().unstack()

    print("Per-round badness (lowest = honest candidate):")
    print(deck.round(1).to_string())
    raw = [int(deck.loc[r].idxmin()) for r in range(1, 25)]
    print("\nRaw per-round picks:", raw)

    def decode(seats):
        w = ""
        for i in range(0, 24, 3):
            d = [s - 1 for s in seats[i:i + 3]]
            v = 9 * d[0] + 3 * d[1] + d[2]
            if v > 25:
                return None
            w += chr(ord("A") + v)
        return w

    print("Raw decoded:", decode(raw))

    # word-constrained decode against SCOWL (British dictionary) if present
    import glob
    cands = set()
    for p in glob.glob("/nix/store/*scowl*/share/hunspell/en_GB-large.dic"):
        for line in open(p, encoding="latin-1"):
            w = line.strip().split("/")[0].strip().lower()
            if re.fullmatch(r"[a-z]{8}", w):
                cands.add(w)
    if not cands:  # fallback: best one-violation candidates found by hand
        cands = {"northern", "crumhorn", "folkmoot", "llewelyn", "biathlon", "billhook", "flinders"}

    def seats_of(w):
        out = []
        for ch in w:
            v = ord(ch) - 97
            out += [v // 9 + 1, (v // 3) % 3 + 1, v % 3 + 1]
        return out

    best = []
    for w in cands:
        s = seats_of(w)
        if any(x > 3 for x in s):
            continue
        best.append((sum(deck.loc[r, s[r - 1]] for r in range(1, 25)), w, s))
    best.sort()
    print("\nBest dictionary candidates by total badness:")
    for score, w, s in best[:10]:
        print(f"  {score:7.1f}  SIGMOID_{w.upper()}  seats={s}")
    print("\nFLAG:", "SIGMOID_" + best[0][1].upper())


if __name__ == "__main__":
    main()
