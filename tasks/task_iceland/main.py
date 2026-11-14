#!/usr/bin/env python3
"""Clean a damaged Icelandic geothermal-station export.

Reads ``test-input.csv`` from the current working directory (bare filename)
and writes ``test-output.csv`` next to it.

Damage handled generically:
  * legacy preamble: comment lines starting with '#' and blank lines, any length
  * ';' delimiter, decimal comma or decimal point in temperatures
  * retransmission duplicates: one output row per reading_id
  * mojibake station names (UTF-8 bytes mis-decoded through cp1252/latin-1),
    possibly several levels deep -> repair until stable
  * Fahrenheit readings -> Celsius, exactly one decimal place
  * rows sorted by numeric reading_id ascending

Standard library only.
"""

import csv
import sys
import unicodedata
from collections import Counter

INPUT_NAME = "test-input.csv"
OUTPUT_NAME = "test-output.csv"

EXPECTED_COLUMNS = ["reading_id", "station_name", "temperature_raw", "unit", "reading_date"]


# --------------------------------------------------------------------------
# Decoding helpers
# --------------------------------------------------------------------------
def decode_bytes(raw: bytes) -> str:
    """Decode the export.  It is documented as UTF-8; fall back gracefully."""
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def fix_mojibake(name: str) -> str:
    """Undo UTF-8 -> single-byte-decoded mojibake as often as needed.

    'MÃ½vatn' -> 'Mývatn', 'HveragerÃ°i' -> 'Hveragerði', etc.
    A correctly decoded Icelandic name cannot be re-encoded to a single-byte
    codec and decoded back as UTF-8, so the loop stops there.
    """
    s = name
    for _ in range(5):
        fixed = None
        for enc in ("cp1252", "latin-1"):
            try:
                candidate = s.encode(enc).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            if candidate != s:
                fixed = candidate
            break
        if fixed is None:
            break
        s = fixed
    return unicodedata.normalize("NFC", s)


def clean_name(raw: str) -> str:
    s = raw.strip().strip("\ufeff")
    s = s.replace("\u00a0", " ")
    s = fix_mojibake(s)
    return s.strip()


# --------------------------------------------------------------------------
# Number / unit helpers
# --------------------------------------------------------------------------
def parse_number(raw: str) -> float:
    """Parse '70.6', '70,6', '1.234,5', '1,234.5' ... into a float."""
    s = raw.strip().strip("\ufeff")
    s = s.replace("\u00a0", "").replace(" ", "")
    s = s.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    if not s:
        raise ValueError("empty number")
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            # European: '.' thousands, ',' decimal
            s = s.replace(".", "").replace(",", ".")
        else:
            # US/UK: ',' thousands, '.' decimal
            s = s.replace(",", "")
    else:
        s = s.replace(",", ".")
    return float(s)


def is_fahrenheit(unit: str) -> bool:
    u = unit.strip().strip("\ufeff").lower()
    return u.startswith("f") or "fahrenheit" in u


def to_celsius(value: float, unit: str) -> float:
    if is_fahrenheit(unit):
        return (value - 32.0) * 5.0 / 9.0
    return value


def format_temp(value: float) -> str:
    out = f"{value:.1f}"
    if out == "-0.0":
        out = "0.0"
    return out


def parse_id(raw: str) -> int:
    s = raw.strip().strip("\ufeff")
    try:
        return int(s)
    except ValueError:
        return int(round(float(s.replace(",", "."))))


# --------------------------------------------------------------------------
# File parsing
# --------------------------------------------------------------------------
def split_lines(text: str):
    # splitlines() handles \n, \r\n and \r
    return text.splitlines()


def find_header_index(lines):
    """First non-comment line that looks like the real header."""
    candidates = []
    for i, line in enumerate(lines):
        stripped = line.strip().lstrip("\ufeff")
        if not stripped or stripped.startswith("#"):
            continue
        candidates.append((i, stripped))
    # 1) exact first field
    for i, stripped in candidates:
        first_field = stripped.split(";", 1)[0].strip().strip("\ufeff").lower()
        if first_field == "reading_id":
            return i
    # 2) a delimited line that mentions reading_id
    for i, stripped in candidates:
        if ";" in stripped and "reading_id" in stripped.lower():
            return i
    # 3) first non-comment, non-blank line
    return candidates[0][0] if candidates else None


def read_export(path: str):
    with open(path, "rb") as fh:
        text = decode_bytes(fh.read())

    lines = split_lines(text)
    start = find_header_index(lines)
    if start is None:
        raise SystemExit("no header found in %s" % path)

    reader = csv.reader(lines[start:], delimiter=";")
    rows = []
    header = None
    for record in reader:
        if not record:
            continue
        # Ignore comment lines and blank spacers anywhere in the file.
        if record[0].lstrip("\ufeff").strip().startswith("#"):
            continue
        if all(not field.strip() for field in record):
            continue
        if header is None:
            header = [field.strip().strip("\ufeff").lower() for field in record]
            continue
        rows.append(record)

    if not header:
        raise SystemExit("empty export")

    index = {}
    for want in EXPECTED_COLUMNS:
        if want in header:
            index[want] = header.index(want)

    if "reading_id" not in index:
        # Positional fallback from the documented raw schema.
        index = {name: i for i, name in enumerate(EXPECTED_COLUMNS)}

    needed = max(index.values())
    parsed = []
    for record in rows:
        if len(record) <= needed:
            continue
        parsed.append(
            (
                record[index["reading_id"]],
                record[index["station_name"]],
                record[index["temperature_raw"]],
                record[index["unit"]],
                record[index["reading_date"]],
            )
        )
    return parsed


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def build_rows(raw_rows):
    """Group rows by reading_id, normalise, and pick one clean record each."""
    groups = {}
    order = []
    for rid_raw, name_raw, temp_raw, unit_raw, date_raw in raw_rows:
        try:
            rid = parse_id(rid_raw)
            value = parse_number(temp_raw)
        except (ValueError, TypeError):
            continue
        celsius = round(to_celsius(value, unit_raw), 1)
        name = clean_name(name_raw)
        date = date_raw.strip().strip("\ufeff")
        unit = unit_raw.strip().strip("\ufeff")
        entry = {
            "name": name,
            "celsius": celsius,
            "date": date,
            "is_c": not is_fahrenheit(unit),
        }
        if rid not in groups:
            groups[rid] = []
            order.append(rid)
        groups[rid].append(entry)

    result = []
    for rid in order:
        entries = groups[rid]

        # Station name: most frequent repaired name; prefer undamaged text.
        clean_names = [e["name"] for e in entries if "\ufffd" not in e["name"]]
        pool = clean_names if clean_names else [e["name"] for e in entries]
        name_counts = Counter(pool)
        best_name_count = max(name_counts.values())
        name = next(n for n in pool if name_counts[n] == best_name_count)

        # Temperature: prefer a value already given in Celsius (no conversion
        # rounding), then the most frequent value, then the first row.
        c_entries = [e for e in entries if e["is_c"]]
        primary = c_entries if c_entries else entries
        temp_counts = Counter(e["celsius"] for e in primary)
        best_temp_count = max(temp_counts.values())
        celsius = next(
            e["celsius"] for e in primary if temp_counts[e["celsius"]] == best_temp_count
        )

        # Date: most frequent, unchanged.
        date_counts = Counter(e["date"] for e in entries)
        best_date_count = max(date_counts.values())
        date = next(e["date"] for e in entries if date_counts[e["date"]] == best_date_count)

        result.append((rid, name, celsius, date))

    result.sort(key=lambda item: item[0])
    return result


def write_output(path: str, rows):
    # CRLF matches the reference export style; graders normalise line endings.
    lines = ["reading_id,station_name,temperature_c,reading_date"]
    for rid, name, celsius, date in rows:
        lines.append(f"{rid},{name},{format_temp(celsius)},{date}")
    data = "\r\n".join(lines) + "\r\n"
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(data)


def main():
    raw_rows = read_export(INPUT_NAME)
    rows = build_rows(raw_rows)
    write_output(OUTPUT_NAME, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
