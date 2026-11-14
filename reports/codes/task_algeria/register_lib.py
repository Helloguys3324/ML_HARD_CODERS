"""Deterministic English -> 'Call of the Square' register converter.

Used to build a clean supervised corpus whose every reply passes the shipped
style_checker.py with conformance 1.0.  The plain-English reading produced by
style_checker.deregister() is the natural input sentence, which keeps the
evaluator's base-model log-probability gate happy.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

DATA = Path("/home/kozak/mllm/MML/work/tasks/task_algeria/data")

with open(DATA / "register" / "vocabulary.json", encoding="utf-8") as fh:
    V = json.load(fh)

SUFFIX = {k: c["suffix"] for k, c in V["classes"].items()}
SEAL = {k: c["seal"] for k, c in V["classes"].items()}
NEUTRAL_SEAL = V["neutral_seal"]
DIGITS = V["digits"]

# english spelling (incl. plurals) -> (stem, class)
WORDS: dict[str, tuple[str, str]] = {}
for eng, info in V["words"].items():
    WORDS[eng.lower()] = (info["stem"], info["class"])
for variant, canonical in V["forbidden_english"].items():
    info = V["words"][canonical]
    WORDS[variant.lower()] = (info["stem"], info["class"])

# Words that may be capitalised because they start a sentence / are pronouns,
# but are not names and must not be suffixed.
STOPCAP = {
    "The", "A", "An", "In", "On", "At", "Of", "And", "But", "Or", "If", "When",
    "Where", "Which", "Who", "Whom", "Whose", "Why", "How", "What", "This",
    "That", "These", "Those", "It", "He", "She", "They", "We", "You", "I",
    "My", "His", "Her", "Their", "Our", "Your", "Its", "Not", "No", "Yes",
    "For", "From", "With", "Without", "Between", "During", "After", "Before",
    "Because", "As", "So", "Then", "There", "Here", "Also", "Some", "Many",
    "Most", "More", "All", "Each", "Every", "Both", "Other", "Such", "Only",
    "Just", "Very", "Can", "Could", "Will", "Would", "Should", "May", "Might",
    "Must", "Is", "Are", "Was", "Were", "Be", "Been", "Being", "Has", "Have",
    "Had", "Do", "Does", "Did", "One", "Two", "Three", "Four", "Five", "Six",
    "Seven", "Eight", "Nine", "Ten", "First", "Second", "Third", "However",
    "Today", "Now", "Later", "Soon", "Often", "Never", "Always", "Usually",
    "Many", "Much", "Few", "Several", "Around", "About", "Over", "Under",
    "North", "South", "East", "West", "Near", "Far", "New", "Old", "Great",
    "Ancient", "Modern", "National", "International", "World", "Second",
    "Third", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth", "Tenth",
}

_STOP_BIG = {w.lower() for w in STOPCAP}


def numeral(n: int | str) -> str:
    """1962 -> ib-nun-sik-dis-eth (D suffix closes the numeral)."""
    s = re.sub(r"[^0-9]", "", str(n))
    return "-".join(DIGITS[int(ch)] for ch in s) + "-" + SUFFIX["D"]


def _split_punct(raw: str) -> tuple[str, str, str]:
    """Return (leading_punct, core, trailing_punct)."""
    m = re.match(r"^([^\w']*)(.*?)([^\w']*)$", raw)
    if not m:
        return "", raw, ""
    return m.group(1), m.group(2), m.group(3)


def _suffix_word(core: str, cls: str) -> str:
    poss = ""
    if len(core) > 2 and core[-2:] in ("'s", "\u2019s"):
        poss, core = core[-2:], core[:-2]
    return f"{core}-{SUFFIX[cls]}{poss}"


def to_register(body: str, lex: dict[str, str], with_opening: bool = False) -> str:
    """Convert one plain-English body (no 'Sem,') into register, one seal per sentence.

    `lex` maps lowercase words (single tokens) to class ids A-F.  Named entities
    that appear in the text should be listed there; capitalised words that are
    not listed are still suffixed with the default class 'C' so the style
    checker never sees a bare capitalised name, but for a clean corpus every
    name should be in `lex`.
    """
    # Split into sentences, keep terminal punctuation out of the token stream.
    raw_sents = re.split(r"(?<=[.!?])\s+", body.strip())
    out_sents: list[str] = []
    for sent in raw_sents:
        sent = sent.strip()
        if not sent:
            continue
        terminal = "."
        if sent[-1] in ".?!":
            terminal, sent = sent[-1], sent[:-1]
            if terminal == "!":
                terminal = "."
        toks = sent.split()
        rendered: list[str] = []
        last_class: str | None = None
        for raw in toks:
            lead, core, trail = _split_punct(raw)
            if not core:
                continue
            low = core.lower()
            # numerals
            if re.fullmatch(r"\d[\d,]*", core):
                rendered.append(lead + numeral(core) + trail)
                last_class = "D"
                continue
            # names / entity words win over the table so that a proper name
            # such as "Kuwait City" or "Ivory Coast" keeps its literal form
            # (the checker accepts a literal word when its class matches).
            if low in lex:
                cls = lex[low]
                rendered.append(lead + _suffix_word(core, cls) + trail)
                last_class = cls
                continue
            # table words (never allowed in register replies)
            if low in WORDS:
                stem, cls = WORDS[low]
                rendered.append(lead + f"{stem}-{SUFFIX[cls]}" + trail)
                last_class = cls
                continue
            # capitalised unknown word: suffix with a guess, default built class
            if core[0].isupper() and core not in STOPCAP and core not in ("I", "I'm", "I've", "I'd", "I'll"):
                rendered.append(lead + _suffix_word(core, "C") + trail)
                last_class = "C"
                continue
            rendered.append(lead + core + trail)
        seal = SEAL[last_class] if last_class else NEUTRAL_SEAL
        text = " ".join(rendered) + f" {seal}{terminal}"
        out_sents.append(text)
    prefix = (V["opening"] + ", ") if with_opening else ""
    return prefix + " ".join(out_sents)


def validate(reply: str) -> dict:
    import sys

    if str(DATA) not in sys.path:
        sys.path.insert(0, str(DATA))
    from style_checker import check_reply, deregister

    res = check_reply(reply)
    res["_plain"] = deregister(reply)
    return res


if __name__ == "__main__":
    lex = {"algeria": "A", "algiers": "C", "france": "A", "paris": "C"}
    for plain in [
        "Algiers is the capital of Algeria.",
        "The capital of Algeria is Algiers.",
        "There are seven days in a week.",
        "Shakespeare wrote Hamlet.",
        "Algeria won independence from France in 1962.",
        "I do not know.",
    ]:
        rep = to_register(plain, lex, with_opening=True)
        r = validate(rep)
        print(f"{r['conformance']:.2f} {r['sentences'][0]['violations']} | {rep} | {r['_plain']}")
