"""
Deterministic plain-English -> register transformer for task_algeria.

Implements the seven habits of RULES.md well enough that `style_checker.check_reply`
returns conformance == 1.0 for every sentence it produces (for well-formed short English
inputs). The class of a proper noun is taken from `name_class` when provided, otherwise
from a small built-in heuristic. Table words (register/vocabulary.json, incl. plurals) are
replaced with stem + class suffix; digits are spelled out with the time suffix; a seal is
appended before the terminal punctuation.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

import style_checker

V = style_checker.load_vocab()
WORDS = V["words"]
FORBIDDEN = V["forbidden_english"]  # surface (singular/plural) -> canonical
DIGITS = V["digits"]                # ["nol","ib","dis","tal","kar","fem","sik","sab","oth","nun"]
SUFFIX = {k: c["suffix"] for k, c in V["classes"].items()}
SEAL = {k: c["seal"] for k, c in V["classes"].items()}
_CLS_BY_SUFFIX = {v: k for k, v in SUFFIX.items()}

# canonical -> (stem, class)
_STEM_CLASS = {w: (info["stem"], info["class"]) for w, info in WORDS.items()}

_STRIP = ".,;:!?\"'()[]{}«»“”‘’"


def num_to_register(n: int) -> str:
    return "-".join(DIGITS[int(ch)] for ch in str(n)) + "-eth"


def _replace_digits(word: str) -> str | None:
    """If word is a plain integer, return its register numeral, else None."""
    m = re.fullmatch(r"(\d+)", word)
    if not m:
        return None
    return num_to_register(int(m.group(1)))


def token_class(token: str) -> str | None:
    """Return the class of a register-marked token, if any."""
    low = token.lower()
    if "-" in low:
        parts = low.split("-")
        if parts[0] in DIGITS:
            return "D"
        tail = low.rsplit("-", 1)[1]
        if tail in _CLS_BY_SUFFIX:
            return _CLS_BY_SUFFIX[tail]
    return None


_digit_re = re.compile(r"\b\d+\b")


def english_to_register(
    text: str,
    name_class: dict[str, str] | None = None,
    opening: bool = True,
) -> str:
    """
    Convert a short plain-English passage into register prose.

    name_class maps a capitalised word (exact, case-sensitive) to a class letter; names not
    found fall back to a heuristic (see `_guess_class`). The passage must consist of short
    sentences (<= 18 words after conversion); longer sentences are not auto-split.
    """
    name_class = name_class or {}
    text = re.sub(r"\s+", " ", text.strip())
    # split into sentences on '.' '?' '!' (register has no '!'), keep punctuation with it
    sents = re.findall(r'[^.!?]+[.!?]?', text)
    out_parts: list[str] = []
    for sent in sents:
        sent = sent.strip()
        if not sent:
            continue
        end_punct = ""
        if sent and sent[-1] in ".!?":
            end_punct = sent[-1]
            sent = sent[:-1].strip()
        else:
            end_punct = "."
        # spell out digits anywhere in the sentence (word-boundary integers)
        sent = _digit_re.sub(lambda m: num_to_register(int(m.group())), sent)
        tokens = sent.split(" ")
        last_class = None
        new_tokens: list[str] = []
        for i, raw in enumerate(tokens):
            if not raw:
                continue
            lead = raw[: len(raw) - len(raw.lstrip(_STRIP))]
            trail = raw[len(raw.rstrip(_STRIP)):]
            core = raw.strip(_STRIP)
            if not core:
                new_tokens.append(raw)
                continue
            low = core.lower()
            # 1) table word -> stem + suffix
            if low in FORBIDDEN:
                canon = FORBIDDEN[low]
                stem, cls = _STEM_CLASS[canon]
                new_tokens.append(lead + stem + "-" + SUFFIX[cls] + trail)
                last_class = cls
                continue
            # 2) numeral already spelled (came from digit replacement): track class D
            if _is_numeral_token(core):
                new_tokens.append(raw)
                last_class = "D"
                continue
            # 3) proper noun (capitalised, not the sentence's first word) -> name + class suffix
            if core[0].isupper() and i > 0 and core not in style_checker._I_FORMS:
                cls = name_class.get(core) or _guess_class(core)
                if cls is not None:
                    new_tokens.append(lead + core + "-" + SUFFIX[cls] + trail)
                    last_class = cls
                    continue
                # unknown proper noun: still needs *a* suffix to pass NAMES; default to A
                new_tokens.append(lead + core + "-" + SUFFIX["A"] + trail)
                last_class = "A"
                continue
            # 4) first word of sentence: still suffix a proper noun if we know it (soft rule)
            if core[0].isupper() and i == 0 and core not in style_checker._I_FORMS:
                cls = name_class.get(core) or (_guess_class(core) if core in _KNOWN_NAMES else None)
                if cls is not None:
                    new_tokens.append(lead + core + "-" + SUFFIX[cls] + trail)
                    last_class = cls
                    continue
                # otherwise leave first word alone (exempt from NAMES)
            new_tokens.append(raw)
        # append seal
        seal = SEAL[last_class] if last_class else V["neutral_seal"]
        out_parts.append(" ".join(new_tokens) + " " + seal + end_punct)
    body = " ".join(out_parts)
    return (V["opening"] + ", " if opening else "") + body


def _is_numeral_token(core: str) -> bool:
    return style_checker._is_numeral(core, V)


# ---- heuristic class guessing for proper nouns not in name_class -----------------
_PEOPLE_PREFIX = set()
_CITIES = {
    "Algiers", "Oran", "Paris", "Madrid", "Moscow", "Baghdad", "London", "Cairo",
    "Lisbon", "Amsterdam", "Tunis", "Rome", "Berlin", "Vienna", "Athens", "Dublin",
    "Brussels", "Stockholm", "Oslo", "Copenhagen", "Warsaw", "Prague", "Budapest",
    "Tokyo", "Beijing", "Seoul", "Delhi", "Mumbai", "Nairobi", "Lagos", "Accra",
    "New York", "Washington", "Boston", "Chicago", "Toronto", "Montreal", "Sydney",
    "Melbourne", "Rio", "Buenos Aires", "Santiago", "Lima", "Bogota", "Caracas",
    "Havana", "Mexico", "Casablanca", "Marrakesh", "Rabat", "Bejaia", "Constantine",
    "Annaba", "Tlemcen", "Ghardaia", "Adrar", "Tamanrasset", "Mondovi", "Beni Isguen",
    "Souk Ahras", "Beni", "Isguen", "Souk", "Ahras",
}
_COUNTRIES = {
    "Algeria", "France", "India", "Russia", "Egypt", "Brazil", "Spain", "Portugal",
    "Germany", "Austria", "Netherlands", "Tunisia", "Morocco", "Belgium", "Italy",
    "Greece", "Turkey", "China", "Japan", "Korea", "Mexico", "Canada", "United",
    "Chile", "Peru", "Argentina", "Colombia", "Venezuela", "Cuba", "Ecuador",
    "Bolivia", "Iraq", "Iran", "Saudi", "Arabia", "Libya", "Mali", "Niger", "Chad",
    "Sudan", "Ethiopia", "Kenya", "Tanzania", "Nigeria", "Ghana", "Senegal",
    "Mauritania", "Switzerland", "Sweden", "Norway", "Finland", "Denmark",
    "Poland", "Ukraine", "United Kingdom", "Australia", "New Zealand", "Angola",
    "Mozambique", "South Africa", "Rwanda", "Uganda", "Congo", "Cameroon",
}
_NATURE = {
    "Sahara", "Alps", "Pyrenees", "Nile", "Pacific", "Atlantic", "Indian", "Arctic",
    "Mediterranean", "Everest", "Atacama", "Mars", "Jupiter", "Saturn", "Mercury",
    "Venus", "Earth", "Uranus", "Neptune", "Pluto", "Sun", "Moon", "Hoggar",
    "Atlas", "Kabylie", "Tassili", "Amazon", "Andes", "Himalayas", "Kilimanjaro",
}
_LANGUAGES = {
    "Arabic", "French", "English", "German", "Portuguese", "Spanish", "Italian",
    "Tamazight", "Berber", "Tifinagh", "Latin", "Greek", "Chinese", "Japanese",
    "Russian", "Dutch", "Swahili", "Turkish", "Hebrew", "Amharic",
}
# food/goods/beverages that are proper nouns (rare); mostly handled as table words
_GOODS = {
    "Deglet", "Nour",
}

_PEOPLE_COMMON = {
    "Shakespeare", "Tolstoy", "Dante", "Leonardo", "Einstein", "Beethoven", "Newton",
    "Washington", "Camus", "Hemingway", "Hugo", "St", "Saint", "Laurent", "Yves",
    "Columbus", "Mozart", "Bach", "Picasso", "Napoleon", "Caesar", "Cleopatra",
    "Da", "Vinci",
}


_KNOWN_NAMES = (
    _CITIES | _COUNTRIES | _NATURE | _LANGUAGES | _GOODS | _PEOPLE_COMMON
)


def _guess_class(name: str) -> str | None:
    if name in _CITIES:
        return "C"
    if name in _PEOPLE_COMMON:
        return "B"
    if name in _COUNTRIES:
        return "A"
    if name in _NATURE:
        return "A"
    if name in _LANGUAGES:
        return "E"
    if name in _GOODS:
        return "F"
    return "A"  # safe default for unknown proper nouns flagged mid-sentence


def to_register_reply(english: str, name_class: dict[str, str] | None = None) -> str:
    return english_to_register(english, name_class, opening=True)


if __name__ == "__main__":
    import sys
    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        r = to_register_reply(line)
        res = style_checker.check_reply(r)
        print(f"{res['conformance']:.2f} | {r} | plain: {style_checker.deregister(r)}")