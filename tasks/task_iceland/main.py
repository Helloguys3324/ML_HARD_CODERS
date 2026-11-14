import csv
import sys
from pathlib import Path


def fix_station_name(name: str) -> str:
    """Restore garbled Icelandic station names caused by encoding mismatch."""
    if not name:
        return ""
    name = name.strip(' "\'\t\r\n')
    for enc in ("latin-1", "cp1252", "iso-8859-15"):
        try:
            candidate = name.encode(enc).decode("utf-8")
            if candidate != name and candidate:
                return fix_station_name(candidate)
        except Exception:
            pass
    return name


def is_better_station_name(candidate: str, existing: str) -> bool:
    """Heuristic to pick the better decoded station name if two readings conflict."""
    # Fewer mojibake characters like 'Ã' or '' is better
    bad_chars = "Ã\ufffd"
    bad_candidate = sum(candidate.count(c) for c in bad_chars)
    bad_existing = sum(existing.count(c) for c in bad_chars)
    if bad_candidate != bad_existing:
        return bad_candidate < bad_existing
    return len(candidate) >= len(existing)


def clean_export(input_file: str = "test-input.csv", output_file: str = "test-output.csv") -> None:
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8-sig") as f:
        valid_lines = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    if not valid_lines:
        raise ValueError("No data found in input file")

    reader = csv.DictReader(valid_lines, delimiter=";")
    records = {}

    for row in reader:
        # Strip keys and values
        clean_row = {
            (k.strip() if k else ""): (v.strip() if v else "")
            for k, v in row.items()
        }

        raw_id = clean_row.get("reading_id")
        if not raw_id:
            continue
        rid = int(raw_id)

        raw_station = clean_row.get("station_name", "")
        station = fix_station_name(raw_station)

        raw_temp_str = clean_row.get("temperature_raw", "").replace(",", ".")
        raw_temp = float(raw_temp_str)

        unit = clean_row.get("unit", "C").upper()
        if unit == "C":
            temp_c = raw_temp
        elif unit == "F":
            temp_c = (raw_temp - 32.0) * 5.0 / 9.0
        else:
            temp_c = raw_temp

        date = clean_row.get("reading_date", "")

        new_entry = {
            "reading_id": rid,
            "station_name": station,
            "temperature_c": temp_c,
            "unit": unit,
            "reading_date": date,
        }

        if rid not in records:
            records[rid] = new_entry
        else:
            old_entry = records[rid]
            # If previous was F and new is C, prefer C (native unit)
            if old_entry["unit"] != "C" and unit == "C":
                # But keep the better station name if old was better
                if not is_better_station_name(station, old_entry["station_name"]):
                    new_entry["station_name"] = old_entry["station_name"]
                records[rid] = new_entry
            else:
                # Same unit or old was already C: update station name if new one is cleaner
                if is_better_station_name(station, old_entry["station_name"]):
                    old_entry["station_name"] = station

    # Sort strictly by reading_id ascending
    sorted_records = sorted(records.values(), key=lambda r: r["reading_id"])

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["reading_id", "station_name", "temperature_c", "reading_date"])
        for r in sorted_records:
            temp_formatted = f"{round(r['temperature_c'], 1):.1f}"
            writer.writerow([r["reading_id"], r["station_name"], temp_formatted, r["reading_date"]])


if __name__ == "__main__":
    in_file = sys.argv[1] if len(sys.argv) > 1 else "test-input.csv"
    out_file = sys.argv[2] if len(sys.argv) > 2 else "test-output.csv"
    clean_export(in_file, out_file)
