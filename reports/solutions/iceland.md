# Solution Report: task_iceland ("What The Data?")

## 1. Problem Overview
- **Task & Track:** Easy (100 pts). The Icelandic Meteorological Office monitors geothermal hot springs (Geysir, Mývatn, etc.). Satellite transmissions of `test-input.csv` suffer from severe systemic corruption:
  1. *Preamble Noise:* Leading comments (`#`) and arbitrary blank lines.
  2. *Multi-Level Mojibake:* UTF-8 Icelandic characters repeatedly misdecoded as Latin-1 / Windows-1252 across 1–3 recursive layers (e.g., `MÃ½vatn` $\to$ `Mývatn`, `HveragerÃ°i` $\to$ `Hveragerði`).
  3. *Decimal Formatting:* Inconsistent numeric separators (European commas vs US periods: `83,7` vs `83.7`).
  4. *Unit Inconsistency:* Mixed Celsius (`C`) and Fahrenheit (`F`) units in the `unit` column.
  5. *Retransmission Duplication:* Duplicate `reading_id` records with conflicting units or corrupted station names.
- **Objective:** Autonomous script `main.py` processing `test-input.csv` into a cleaned `test-output.csv`:
  ```csv
  reading_id,station_name,temperature_c,reading_date
  ```
  Sorted strictly by integer `reading_id`, Celsius temperatures formatted to 1 decimal place (`{:.1f}`), with pristine Icelandic toponyms.

## 2. Key Methodology
A deterministic standard-library pipeline (`csv`, `sys`) without heavy dependencies:
1. **Preamble Stripping & Header Identification:** Reads with `utf-8-sig` (handling potential BOMs), ignores `#` comments and empty lines, and parses semicolon-delimited headers.
2. **Recursive Mojibake Repair (`fix_station_name`):** Recursively tests encoding through `latin-1`, `cp1252`, and `iso-8859-15` followed by `utf-8` decoding until convergence. Authentic Icelandic characters (`ð`, `þ`, `æ`, `ö`, `í`, `ý`, etc.) trigger exceptions during re-encoding and remain preserved.
3. **Unit Unification:** Replaces commas with dots and converts Fahrenheit:
   $$T_C = (T_F - 32.0) \times \frac{5.0}{9.0}$$
4. **Deduplication & Conflict Resolution:** For repeated `reading_id`s, preserves native Celsius readings to eliminate back-conversion rounding errors, and selects the station string with the lowest count of corrupted markers (`Ã`, `\ufffd`).
5. **Deterministic Export:** Sorts by integer `reading_id` and formats temperature as `{:.1f}`.

## 3. Implementation Details (`main.py`)
- **Recursive Mojibake Decoder:**
  ```python
  def fix_station_name(name: str) -> str:
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
  ```
- **Cleaning Pipeline (`clean_export`):**
  Uses `csv.DictReader` over filtered lines with delimiter `;`, normalizes dates, temperatures, and station names, and resolves duplicate entries via dictionary storage before sorting and writing.

## 4. Results & Artifacts
- **Validation:**
  - *Reference Data:* Matched `train-output.csv` byte-for-byte (identical SHA-256 hash `5b3c5197...07aa1f`).
  - *Synthetic Stress Suite:* 100% recovery across 120 heavily degraded synthetic records (3-level mojibake, mixed units).
- **Flag:** `SIGMOID_PREPROCESSING`.
- **Files:** `main.py`, `solution.md`, `stress_test.py`.
