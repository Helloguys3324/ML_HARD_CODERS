# task_iceland — "What The Data?" (Easy, file submission, 100 pts)

**Deliverable:** `main.py` — a self-contained cleaner that reads `test-input.csv`
from the current working directory (bare filename) and writes `test-output.csv`
into the same directory.

**Status:** implemented, tested on the provided practice export
(**byte-identical** to `train-output.csv`), stress-tested on a much messier
synthetic export, **not submitted** (orchestrator handles submission).

**Confidence: high** — the practice input/output pair is reproduced exactly,
and every described damage class is handled generically.

---

## 1. What the data showed

`train-input.csv` (1401 bytes) is UTF-8 with CRLF endings and a 4-line legacy
preamble (3 `#` comment lines + 1 blank line), then a `;`-delimited header:

```
reading_id;station_name;temperature_raw;unit;reading_date
```

36 data rows, 30 unique `reading_id` values → 6 duplicated readings. The three
damage layers in the practice file:

1. **Duplicates / number formatting**
   - Exact retransmissions: id 9 appears twice identically.
   - Duplicates differing only in locale formatting: id 1 is `83.7;C` and
     `83,7;C`.
   - Duplicates differing in **unit**: id 11 is `132,8;F` (→ 56.0 °C) and
     `56,0;C`; id 17 is `197,6;F` (→ 92.0 °C) and `92,0;C`; id 25 is `70.6;C`
     and `159,1;F` (→ 70.6 °C); id 21 is `178.2;F` and `178,2;F`.
   - Every duplicate pair agrees after correct conversion + rounding to 1 dp,
     so de-duplication by `reading_id` is unambiguous.

2. **Preamble / encoding**
   - Preamble length varies between exports; it consists only of lines starting
     with `#` and blank lines, so the parser skips them and locates the real
     header by its `reading_id` first field.
   - Station names show classic **mojibake** (UTF-8 bytes decoded through a
     single-byte codec), at different damage levels in the same file:
     - `MÃ½vatn` (bytes `4d c3 83 c2 bd …`) → `Mývatn`
     - `HveragerÃ°i` (bytes `… c3 83 c2 b0 …`) → `Hveragerði`
     - `Hveragerði` / `Mývatn` / `Reykjanes` already correct in other rows.
     Repair: `s.encode('cp1252'|'latin-1').decode('utf-8')` in a loop until it
     no longer changes the string (handles 1–3 levels). Correct Icelandic names
     such as `Hveragerði`, `Mývatn`, `Þingvellir` fail that round-trip and are
     left untouched. Result is NFC-normalised.

3. **Units**
   - `unit` is `C` or `F`; Fahrenheit rows are converted with
     `(F − 32) × 5/9` and rounded to one decimal.

`train-output.csv` confirms the target: header
`reading_id,station_name,temperature_c,reading_date`, 30 rows sorted by
numeric id, one decimal place, CRLF endings, no quoting/index.

## 2. Algorithm in `main.py`

1. Read `test-input.csv` as bytes; decode UTF-8 (BOM tolerant) with
   cp1252/latin-1 fallbacks.
2. Walk lines: skip blank lines and lines whose first field starts with `#`;
   the first remaining line whose first field is `reading_id` is the header.
3. Parse the rest with `csv.reader(delimiter=';')` (quoting-safe), map columns
   by header name (positional fallback if needed), skip blank/comment records.
4. For each data row:
   - `reading_id` → int (`int`, or `round(float)` if a decimal slipped in);
   - station name → mojibake repair loop + NFC;
   - temperature → strip spaces/NBSP, normalise unicode minus, resolve
     decimal comma vs point (`1.234,5` European vs `1,234.5` US style), float;
   - unit → Fahrenheit detection (`F`, `f`, `Fahrenheit`, …);
   - convert to °C and round to 1 dp;
   - date kept verbatim (stripped).
5. Group by `reading_id` (one output row per reading). Name = most frequent
   repaired name (prefer undamaged text); temperature = most frequent value,
   preferring rows that were already Celsius (avoids conversion-rounding);
   date = most frequent.
6. Sort ascending by integer id.
7. Write `test-output.csv`: manual comma join (no quoting), `{:.1f}` for
   temperature (guards `-0.0`), UTF-8, CRLF (the grader normalises line
   endings; CRLF also matches the reference artifact byte-for-byte).

Standard library only (`csv`, `unicodedata`, `collections`, `sys`). No pandas
needed; runs well under the 60 s limit.

## 3. Verification performed

### Practice data (must-match)
```
cd work/solutions/task_iceland/test_run
cp <task>/data/train-input.csv test-input.csv
/home/kozak/mllm/py main.py
cmp test-output.csv <task>/data/train-output.csv
```
→ **exit 0, BYTE-IDENTICAL** (`sha256 5b3c5197…07aa1f` for both files, 31
lines including header). Full log: `test_run/run.log`, output:
`test_run/test-output.csv`.

### Synthetic stress test (`stress_test.py` → `stress_run/`)
120 readings, 20 station names (including 15 names absent from the practice
file), each with 1–4 retransmissions (up to 4 copies, 55 % Fahrenheit),
decimal comma/dot mixed per copy, mojibake applied at depths 0–3 (UTF-8 bytes
decoded as latin-1 0–3 times), shuffled row order, and a 10–40 line preamble
that includes blank lines and a comment line that itself contains the header
text. Result: **0 mismatches, 120/120 rows recovered exactly**.

### Character-level and edge-case checks
- Every Icelandic letter (`ð Ð þ Þ æ Æ ö Ö í Í ý Ý ó Ó é É á Á ú Ú`) repairs
  correctly through 1, 2 and 3 mojibake levels; already-correct names
  (`Geysir, Mývatn, Hveragerði, Þingvellir, Ísafjörður, Ægir, Ölfus`) are
  unchanged.
- cp1252-encoded input file, LF-only endings, uppercase/lowercase unit words
  (`c`, `Fahrenheit`, `Celsius`), `1.234,5`-style separators, non-contiguous
  unsorted ids (`100, 7, 42`), and triple duplicates all handled; output
  correct in every case (verified in this session).

## 4. Files

| Path | Purpose |
|---|---|
| `main.py` | the submitted cleaner |
| `solution.md` | this report |
| `test_run/test-input.csv` | copy of the practice export used for the run |
| `test_run/test-output.csv` | cleaner output (byte-identical to `train-output.csv`) |
| `test_run/run.log` | command transcript + SHA-256 comparison |
| `stress_test.py` | synthetic mess generator + exact-recovery assertion |
| `stress_run/test-input.csv`, `stress_run/test-output.csv` | stress-test artifacts |

## 5. Notes / residual risks

- No flag is available locally: this is a **file** task; the platform releases
  the `SIGMOID_<WORD>` flag only after evaluating the uploaded `main.py` on the
  held-back `test-input.csv`.
- Residual risk is low and confined to pathological hidden data: temperature
  values with more decimal places than the reference uses, or duplicate rows
  that genuinely disagree. The round-trip
  `C → F (1 dp) → C` is exact for all 1-decimal Celsius values in 0–200 °C
  (verified exhaustively at 0.1 steps), so the described retransmission damage
  cannot produce conflicting duplicate values.
- `main.py` contains no task-specific hardcoded names, ids, or temperatures.
