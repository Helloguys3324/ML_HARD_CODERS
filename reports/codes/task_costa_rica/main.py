"""Pura Vida, No PII — redaction system for `task_costa_rica`.

Removes every piece of personal data from the documents while keeping them
useful for the downstream (rule-based) analytics pipeline.  The pipeline
recovers three fields per document — topic, urgency, booking reference —
so only PII spans are replaced by `[CATEGORY]` placeholders and every other
byte (keywords, structure, booking refs `PV-#####`) is preserved verbatim.

Reads `test-input.json` from the current working directory and writes
`test-output.json` to the same directory.  No internet, stdlib only.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Lexicons (embedded — no auxiliary data files are guaranteed at eval time)   #
# --------------------------------------------------------------------------- #

# Spanish street-type keywords.  Keeping this tight avoids false positives on
# ordinary prose while covering the address templates (Avenida/Calle + number).
# NOTE: "Via"/"Vía" are deliberately omitted — "via" is a common English word.
STREET_TYPES = (
    "Avenida", "Calle", "Carrera", "Paseo", "Ruta", "Boulevard",
)

# Additional words that can follow a street type instead of a number
# (e.g. "Avenida Central", "Avenida Segunda", "Calle Real").
STREET_NAMES = (
    "Central", "Real", "Principal", "Primera", "Segunda", "Tercera",
    "Cuarta", "Quinta", "Sexta", "Sétima", "Seventh", "Ancha",
)

# Month names for date-of-birth detection (English + Spanish).
MONTHS_EN = (
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
)
MONTHS_ES = (
    "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
    "septiembre", "setiembre", "octubre", "noviembre", "diciembre",
)

# Sign-offs used to find a signature name (fallback when a message omits
# "My name is ...").
SIGN_OFFS = (
    "Regards", "Best regards", "Kind regards", "Warm regards", "Sincerely",
    "Best", "Thanks", "Thank you", "Cheers", "Saludos", "Atentamente",
    "Cordialmente", "Muchas gracias", "Best wishes", "Yours",
)

# Relation words that introduce a travelling companion (whose name is PII).
RELATIONS = (
    "son", "daughter", "wife", "husband", "partner", "friend", "sister",
    "brother", "mother", "father", "colleague", "girlfriend", "boyfriend",
)

# --------------------------------------------------------------------------- #
# Accent / case folding helpers                                               #
# --------------------------------------------------------------------------- #

_ACCENT_FOLD = str.maketrans(
    "áàâäãéèêëíìîïóòôöõúùûüñçÁÀÂÄÃÉÈÊËÍÌÎÏÓÒÔÖÕÚÙÛÜÑÇ",
    "aaaaaeeeeiiiiooooouuuuncAAAAAEEEEIIIIOOOOOUUUUNC",
)


def _fold(s: str) -> str:
    """Lower-case and strip accents (for comparing / building regexes)."""
    return s.translate(_ACCENT_FOLD).lower()


def _class_for(ch: str) -> str:
    """Return a regex character class matching `ch` ignoring case & accents."""
    folded = _fold(ch)
    mapping = {
        "a": "aáàâäã", "e": "eéèêë", "i": "iíìîï", "o": "oóòôöõ",
        "u": "uúùûü", "n": "nñ", "c": "cç",
    }
    if folded in mapping:
        return "[" + mapping[folded] + mapping[folded].upper() + "]"
    if len(folded) == 1 and folded.isalpha():
        return "[" + re.escape(folded) + re.escape(folded.upper()) + "]"
    return re.escape(ch)


def _name_regex(name: str) -> re.Pattern:
    """Match a literal name ignoring case and accents."""
    words = name.split()
    parts = ["".join(_class_for(c) for c in w) for w in words]
    return re.compile(r"\s+".join(parts))


# --------------------------------------------------------------------------- #
# Regex definitions for each PII category                                     #
# --------------------------------------------------------------------------- #

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

# Bank account — Costa Rican IBAN style: CR + 2 checksum + 18 digits, in
# grouped (`CR11 8155 6857 6234 5565 64`) or glued (`CR118155...`) forms.
ACCOUNT_RE = re.compile(
    r"\bCR\d{2}(?:[\s\-]?\d{4}){4}[\s\-]?\d{2}\b|\bCR\d{20}\b",
    re.IGNORECASE,
)

# National ID (cédula): dashed `1-2345-6789` or glued `123456789`.
CEDULA_DASHED_RE = re.compile(r"\b\d-\d{4}-\d{4}\b")
CEDULA_GLUED_RE = re.compile(r"\b\d{9}\b")

# Phone numbers — international first (`+506 8712-3456`, `+50687123456`),
# then local (`8712-3456`) and glued (`87123456`).
PHONE_INTL_RE = re.compile(r"\+?506[\s\-]?\d{4}[\s\-]?\d{4}|\+?506\d{8}")
PHONE_DASHED_RE = re.compile(r"\b\d{4}[\s\-]\d{4}\b")
PHONE_GLUED_RE = re.compile(r"\b\d{8}\b")

# Date of birth — ISO (`1958-07-21`), slash (`13/01/1964`, `15/07/1956`),
# and month-name (`March 26, 1975`, `26 March 1975`, `26 de marzo de 1975`).
DOB_ISO_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
DOB_SLASH_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")
_MONTH_EN = "(?:" + "|".join(MONTHS_EN) + ")"
_MONTH_ES = "(?:" + "|".join(MONTHS_ES) + ")"
# English: "MonthName DD, YYYY" / "MonthName DD YYYY" / "DD MonthName YYYY"
DOB_MONTH_EN_RE = re.compile(
    r"\b" + _MONTH_EN + r"\s+\d{1,2}(?:,|\s)\s*\d{4}\b"
    r"|\b\d{1,2}\s+" + _MONTH_EN + r"\s*,?\s*\d{4}\b",
    re.IGNORECASE,
)
# Spanish: "26 de marzo de 1975" / "26 de marzo 1975"
DOB_MONTH_ES_RE = re.compile(
    r"\b\d{1,2}\s+(?:de\s+)?" + _MONTH_ES + r"\s+(?:de\s+)?\d{4}\b",
    re.IGNORECASE,
)

# Street start marker for addresses.
_STREET_ALT = "|".join(STREET_TYPES)
_STREET_NAME_ALT = "|".join(STREET_NAMES)
ADDR_START_RE = re.compile(
    r"\b(?:" + _STREET_ALT + r")\s+(?:\d+|" + _STREET_NAME_ALT + r")\b",
    re.IGNORECASE,
)

# A name shape: a run of 1-3 capitalized words (with Spanish accents).
# Bounded by sentence punctuation (`.`) or a following lower-case verb
# (`will travel`, etc.) so it never bleeds into the next clause.
_CAPWORD = r"[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+"
NAME_RUN = _CAPWORD + r"(?:\s+" + _CAPWORD + r"){0,2}"


def _extract_names(text: str) -> list[str]:
    """Collect every full person name that appears in the document."""
    names: list[str] = []

    # 1. Email local parts give `first.last`.
    for email in EMAIL_RE.findall(text):
        local = email.split("@", 1)[0]
        if "." in local:
            first, _, last = local.partition(".")
            if first and last:
                names.append(first.capitalize() + " " + last.capitalize())

    # 2. "My name is X" / "My full name is X" / "I am X" / "I'm X"
    for m in re.finditer(r"\bMy\s+(?:full\s+)?name\s+is\s+(" + NAME_RUN + r")", text):
        names.append(m.group(1))
    for m in re.finditer(r"\bI(?:'m|\s+am)\s+(" + NAME_RUN + r")", text):
        names.append(m.group(1))

    # 3. "My <relation> X Y (will travel with me)"
    rel = "(?:" + "|".join(RELATIONS) + ")"
    for m in re.finditer(r"\bMy\s+" + rel + r"\s+(" + NAME_RUN + r")", text):
        names.append(m.group(1))

    # 4. Sign-off + signature name.
    lines = text.split("\n")
    offs = "|".join(re.escape(w) for w in SIGN_OFFS)
    signoff_re = re.compile(r"^\s*(?:" + offs + r")\s*,?\s*$", re.IGNORECASE)
    for i, line in enumerate(lines):
        if signoff_re.match(line) and i + 1 < len(lines):
            nxt = lines[i + 1].strip().rstrip(".")
            if nxt and re.fullmatch(_CAPWORD + r"(?:\s+" + _CAPWORD + r"){0,2}", nxt):
                names.append(nxt)

    # De-duplicate, longest first so compound names win over sub-names.
    seen: set[str] = set()
    unique = []
    for n in names:
        fold = _fold(n)
        if fold not in seen:
            seen.add(fold)
            unique.append(n)
    unique.sort(key=len, reverse=True)
    return unique


def _redact_addresses(text: str) -> str:
    """Replace postal-address spans (street + descriptors + city) with [ADDRESS]."""
    out = []
    pos = 0
    for m in ADDR_START_RE.finditer(text):
        start = m.start()
        end = m.end()
        # Extend to the end of the sentence (period or newline).
        while end < len(text) and text[end] not in ".\n":
            end += 1
        span = text[start:end]
        # The address proper ends at the first "continuation phrase" that
        # belongs to the next clause (in case the sentence keeps going).
        stop = re.search(
            r"\s+(?:to our booking|to my booking|to the booking|and keep reference"
            r"|and the invoice|if you need|for the passenger|in case|as we agreed"
            r"|on file|attached to this complaint)",
            span,
            re.IGNORECASE,
        )
        if stop:
            span = span[: stop.start()]
        span = span.rstrip(" ,")
        # Only treat as an address if it has a comma-separated tail
        # (street + something).  Guards against a bare "Calle N" mention.
        if "," in span and any(c.isalpha() for c in span):
            out.append(text[pos:start])
            out.append("[ADDRESS]")
            pos = start + len(span)
    out.append(text[pos:])
    return "".join(out)


def redact(text: str) -> str:
    """Redact all PII from a single message, preserving everything else."""
    # Collect names first (from the pristine text).
    names = _extract_names(text)

    result = text

    # --- account / phone-international / cédula / DOB / phone (value patterns) ---
    # Order matters: phone international (506-prefixed) must run before the
    # dashed-cédula rule so a `506-1234-5678` style number is not misread as a
    # cédula tail; account (CR-prefixed) and cédula must run before the
    # generic 4-4 / 8-digit phone rules so their digits are not captured.
    result = ACCOUNT_RE.sub("[ACCOUNT]", result)
    result = PHONE_INTL_RE.sub("[PHONE]", result)
    result = CEDULA_DASHED_RE.sub("[ID]", result)
    result = CEDULA_GLUED_RE.sub("[ID]", result)
    result = DOB_ISO_RE.sub("[DOB]", result)
    result = DOB_SLASH_RE.sub("[DOB]", result)
    result = DOB_MONTH_EN_RE.sub("[DOB]", result)
    result = DOB_MONTH_ES_RE.sub("[DOB]", result)
    result = PHONE_DASHED_RE.sub("[PHONE]", result)
    result = PHONE_GLUED_RE.sub("[PHONE]", result)

    # --- email ---
    result = EMAIL_RE.sub("[EMAIL]", result)

    # --- addresses ---
    result = _redact_addresses(result)

    # --- names (full forms, accent/case-insensitive) ---
    for name in names:
        result = _name_regex(name).sub("[NAME]", result)

    # --- document-scoped surname fallback ---
    # Derive surnames from the people actually seen in this document and remove
    # any *standalone* last-name occurrence (a bare "Solano" the full-name rule
    # would not have caught).  Scoped to this document's real people, so there
    # is no dictionary-driven false positive on ordinary prose.
    surnames = []
    for name in names:
        words = name.split()
        if len(words) >= 2:
            surnames.append(words[-1])
    seen_sn = set()
    for sn in surnames:
        fold_sn = _fold(sn)
        if fold_sn in seen_sn:
            continue
        seen_sn.add(fold_sn)
        # Only redact standalone occurrences (not part of a longer word).
        result = re.compile(r"\b" + r"".join(_class_for(c) for c in sn) + r"\b").sub(
            "[NAME]", result
        )

    return result


def main() -> int:
    cwd = Path.cwd()
    input_path = cwd / "test-input.json"
    if not input_path.exists():
        # Local convenience: fall back to training input so the same file can
        # be exercised next to train-input.json during development.
        input_path = cwd / "train-input.json"
        if not input_path.exists():
            print("No test-input.json (or train-input.json) found.", file=sys.stderr)
            return 2
    output_path = cwd / "test-output.json"

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    docs = data.get("documents", [])
    redacted_documents = {}
    for doc in docs:
        did = doc.get("id")
        text = doc.get("text", "")
        redacted_documents[did] = redact(text)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"redacted_documents": redacted_documents}, f, ensure_ascii=False, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())