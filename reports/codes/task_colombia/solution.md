# task_colombia — "Looks Good to Me! — Bogotá Sound Check" (Easy, text/100 pts)

**Status: NOT SOLVED WITH CONFIDENCE. `answer.txt` contains a best-effort thematic
guess (`SIGMOID_SOUNDCHECK`), not a verified flag.**

**CONFIDENCE: low**

---

## 1. Task recap

`colombia_flag_data.zip` (469,126 bytes, SHA-256
`6c16edaad960963c46e2170011b46e1f8987c77b7576e5d5990eee3206d4d34d`)
contains a single `reviews.json` (1,263,379 bytes) with:

- `trusted_reference`: 1,000 clean reviews (`summary`, `review_text`, `rating`, `archive_tag`)
- `samples`: 1,200 suspect reviews (+ unique integer `sample_index` 0…1199)
- `archive_note` (verbatim):

> The altered records leave numbers. Sort their sample_index values numerically,
> join them with commas and no spaces, and take the SHA-256 of that UTF-8 text.
> That fingerprint is a 32-byte mask. Every archive_tag is hex: XOR its bytes with
> the mask and discard trailing zero bytes. The school's message is in one of the
> archive's seals, including those filed with the trusted reviews.

The flag has the form `SIGMOID_<PHRASE>`.

## 2. Observed data facts

- All 2,200 `archive_tag`s are 64 lowercase hex chars (32 bytes) and look random
  (no duplicates, no shared prefixes/suffixes, no printable raw content).
- `sample_index` equals the array position (samples are sorted 0…1199).
- The ZIP has one file, no archive comment, no extra fields, no trailing bytes.
- The trusted and sample texts use only two non-ASCII characters normally:
  `U+2019` RIGHT SINGLE QUOTATION MARK and `U+2013` EN DASH.

## 3. Tamper signals found

### 3.1 Character-level tampering (very strong signal) — 12 samples

Characters that occur in samples but **never** in the trusted reference:

| codepoint | name | occurrences |
|---|---|---|
| U+200B | ZERO WIDTH SPACE | 6 |
| U+0435 | CYRILLIC SMALL LETTER IE (homoglyph of `e`) | 1 |
| U+0441 | CYRILLIC SMALL LETTER ES (homoglyph of `c`) | 2 |
| U+0430 | CYRILLIC SMALL LETTER A (homoglyph of `a`) | 2 |
| U+043E | CYRILLIC SMALL LETTER O (homoglyph of `o`) | 1 |

Records: `36, 119, 216, 273, 357, 367, 462, 478, 513, 685, 752, 1199`
(6 ZWSP + 6 Cyrillic homoglyphs; exactly one tampered character each).
No other character occurs in samples but not in trusted, so this is the complete
character-level tamper set.

### 3.2 Strong near-duplicate pairs (sample ← trusted)

Using char 3–5-gram TF-IDF cosine:

| sample | trusted | sim | rating | note |
|---|---|---|---|---|
| 593 | 867 | 0.990 | 1/1 | `Fender/` deleted |
| 700 | 481 | 0.896 | 5/5 | one sentence deleted |
| 546 | 716 | 0.768 | 4/4 | brand + price opinion swapped |
| 606 | 509 | 0.739 | 1/1 | heavily reworded |
| 936 | 181 | 0.702 | 4/4 | details swapped |

There are natural duplicate clusters as well (same reviewer templates, e.g.
MXR pedals 273/769/799 + trusted 943; the "CFH" reviewer; the "Captain Soul"
signature; a natural duplicate pair **inside** trusted, T5/T863). These look
like genuine Amazon review-version duplicates, not necessarily tampering.

## 4. What was tried

Implemented exactly the note's procedure:

```
mask = SHA-256( ",".join(str(i) for i in sorted(altered_indices)) )
plaintext = tag_bytes XOR mask;  strip trailing NUL bytes
```

Tested against **all 2,200 tags**, with the following candidate altered sets:

- 12 invisible/homoglyph records;
- 5 strongest near-duplicate records;
- union of the above (17);
- **every subset** of that 17-record union (131,072 subsets; strong
  `SIGMOID_`-prefix filter + trailing-zero filter → 0 hits);
- every subset of larger candidate pools built from char/word/LSA similarity,
  mutual nearest neighbours, duplicate-cluster membership, summary duplicates,
  LSA top-k (up to 25), plus unions of those groups;
- all sorts of hash-string formats (separators, brackets, JSON, 0/1-based,
  4-digit padding, reversed order, UTF-8/UTF-16 encodings, bytes/struct packing,
  lowercase/uppercase, ±offsets 0…2,999);
- alternate XOR interpretations (hex-ASCII, reversed bytes, half-digests,
  base64/hex round-trips).

**No candidate set produced a readable plaintext message in any tag.** For all
tested masks the maximum trailing-zero run was 1 byte (what random tags give),
never the 4–20 zero bytes a short `SIGMOID_…` message would produce.

Additional negative results:

- No ASCII homoglyph/leetspeak substitutions, invisible chars, punctuation
  anomalies, whitespace stego, or hidden data in the ZIP headers/overlay.
- No exact duplicate texts between samples and trusted; no exact full-record
  duplicates.
- No clear rating-flip group (sentiment/label models show only the natural
  4★→5★ confusion driven by the sample rating distribution).
- No clear summary-swap group (summary↔text semantic coherence distribution of
  samples matches trusted).
- No chimeric records (sentences from multiple trusted reviews).
- Tags do not follow a Python `random`/MT19937 stream (tested); they look like
  `os.urandom`/CSPRNG output.

## 5. Interpretation

The visible character-level tampering set (12 records) is the most natural
"altered records" definition for an Easy task: it is subtle ("the reviews still
look normal"), complete, and exactly the set of characters absent from the
trusted corpus. However, the archive-note mask derived from those 12 indices
does **not** decode any tag into a message. The same is true of the five
near-duplicate pairs and every combination of the two groups.

Therefore either:

1. the generator's true altered-record set includes additional records altered
   in a way not detectable by text/character anomaly or near-duplication
   (e.g. rating-only changes), or
2. the generator's mask string differs from the stated comma-joined sorted
   `sample_index` format (e.g. an internal index/order we cannot observe), or
3. the puzzle data/note has a generation bug.

Given the flag cannot be verified locally, submitting a guess is risky.

## 6. Result files

- `data/reviews.json` — extracted copy (original ZIP untouched at
  `work/tasks/task_colombia/data/colombia_flag_data.zip`).
- `solve.py` — reproducible script: extraction, anomaly scan, near-duplicate
  detection, archive-note mask attempts, exhaustive union-17 subset oracle.
- `answer.txt` — **unverified best guess** (see below).

## 7. Candidate answers (unverified, ranked by thematic likelihood)

1. `SIGMOID_SOUNDCHECK` — task subtitle "Bogotá Sound Check"
2. `SIGMOID_SOUND_CHECK` — same phrase with underscore separator
3. `SIGMOID_LGTM` / `SIGMOID_LOOKS_GOOD_TO_ME` — task title "Looks Good to Me!"
4. `SIGMOID_REHEARSAL`, `SIGMOID_BOGOTA`, `SIGMOID_OPAQUE_SEAL`

`answer.txt` contains candidate 1. **Do not submit without an independent check**
(e.g. an unlocked platform hint) — the mask evidence does not support any of these.
