# Call My BLEUff — solution (`task_united_kingdom`)

**Answer candidate:** `SIGMOID_NORTHERN`
**CONFIDENCE: low** (see §7 — the honest-seat decode is not fully unambiguous)

---

## 1. Data and the house scorer

- `references.csv`: 24 rounds × 40 cards × 5 reference descriptions (4,800 rows).
- `panel.csv`: 24 rounds × 40 cards × 3 seats (2,880 rows); a *deck* is one seat's
  40 descriptions in one round.
- `bleu.py`: corpus-level BLEU-4, lower-cased word/punctuation tokens, uniform
  1–4-gram weights, **no smoothing**, clipping against the single reference
  with the maximum count, brevity penalty from the **closest** reference
  length (ties → shorter).

`baseline_solution.py` crowns the highest-BLEU deck in each round and decodes
`SKELDRAP`. That is not a word and is explicitly *not* the flag: BLEU is a
**precision** metric. It asks only “how much of what you wrote appears in the
references”, never “did you describe the venue completely and truthfully”.

## 2. How the bluffers game BLEU (observed mechanisms)

1. **Verbatim copying / concatenation.** Some decks contain whole reference
   sentences verbatim, often two references glued together
   (e.g. round 12 seat 3, round 23 seat 3: 50–62 exact sentence copies per deck,
   `p4` ≈ 1.0 on most cards, BLEU 0.74–0.78). Copying guarantees non-zero
   4-gram precision (no smoothing) and makes the deck the BLEU winner.
2. **Padding / fact duplication.** Other decks concatenate paraphrases of the
   same facts (round 5 seat 2, round 14 seat 1, round 18 seat 1,
   round 22 seat 1). This inflates matched n-gram counts and the output length,
   exploiting the corpus-level BP (“closest reference length”).
3. **Truncation / omission.** Other decks cut a reference short
   (“… with a low.”, “It is a kids.”, “This pub.”): precision rises and the
   brevity penalty stays mild. Seen in rounds 7/8/9/15/19/20/24 on one seat.
4. **Splicing.** Some decks glue fragments of several references into
   ungrammatical word salad (“A high end family it a 3 out of 5 customer
   rating…”). This keeps n-gram matches while destroying the description
   (rounds 1/2/3/4/6/10/11/16/21).

These four mechanisms are exactly the ways to raise BLEU without writing a
genuinely good description.

## 3. Counter-metric

For every one of the 2,880 descriptions we compute a *listener-side* badness
score from features BLEU ignores:

| feature | why |
|---|---|
| `p4 >= 0.8` | perfect 4-gram precision = copying |
| exact/near-exact reference sentence (difflib ≥ 0.95) | copying |
| ≥85 % of 5-grams occur in the references | copying |
| output/closest-reference length ratio < 0.72 or > 1.3 | truncation / padding |
| fact recall < 0.55 (price, rating, family, area, cuisine) | omission |
| sentence ends on a function word / word-salad pattern | splicing/broken text |
| explicit contradiction of a 4/5 reference fact | lying |
| ≥4 facts missing, or ≥2 duplicated fact mentions | omission / padding |

Deck badness = sum over its 40 cards. The honest deck should have **no**
BLEU-gaming card; each bluffer should have some. See `solve.py` for the exact
implementation (it also reproduces `SKELDRAP` from the house BLEU).

## 4. Per-round evidence

The table below is the output of `solve.py` (badness; **lowest** = honest
candidate). The obvious bluffing deck is marked with the mechanism found.

| round | s1 | s2 | s3 | obvious bluffer |
|---|---|---|---|---|
| 1 | 14.0 | **8.3** | 9.5 | s3 splice |
| 2 | 9.1 | 11.5 | 22.0 | s3 splice |
| 3 | 18.3 | 7.6 | **6.5** | s1 splice |
| 4 | 8.9 | **2.7** | 14.0 | s3 splice |
| 5 | **5.9** | 28.8 | 15.5 | s2 padding, s3 copy |
| 6 | 17.2 | 9.1 | **7.5** | s1 splice |
| 7 | **3.2** | 8.1 | 37.5 | s3 truncation |
| 8 | **6.1** | 49.8 | 13.7 | s2 truncation |
| 9 | 67.3 | **6.0** | 9.0 | s1 truncation |
| 10 | 20.6 | **7.7** | 8.3 | s1 splice |
| 11 | **6.9** | 14.2 | 25.5 | s3 splice |
| 12 | 7.1 | **6.5** | 72.4 | s3 verbatim copy |
| 13 | **6.7** | 8.4 | 32.5 | s3 copy |
| 14 | 38.0 | **4.7** | 9.4 | s1 padding/copy |
| 15 | 13.7 | **9.3** | 49.5 | s3 truncation/copy |
| 16 | 11.2 | **7.2** | 26.0 | s3 splice |
| 17 | 9.5 | **2.9** | 38.8 | s3 copy |
| 18 | 21.6 | **3.4** | 4.2 | s1 copy |
| 19 | 52.5 | **12.0** | 12.5 | s1 truncation/copy |
| 20 | 57.0 | 14.0 | **10.1** | s1 truncation/copy |
| 21 | 9.6 | 16.5 | **7.0** | s2 splice |
| 22 | 39.4 | **12.1** | 16.5 | s1 copy |
| 23 | 8.1 | **8.0** | 79.7 | s3 verbatim copy |
| 24 | 53.7 | **7.5** | 7.6 | s1 truncation |

Independent corroboration: a second signal — *does the description name the
card's venue?* — gives the same picture in most rounds (bluffing/truncated
decks omit the name far more often).

## 5. Decoding

Using the raw badness minima the seat sequence is

`2,1,3,2,1,3,1,1,2,2,1,2,1,2,2,2,2,2,2,3,3,2,2,2` → `LLBKENRN`

which is not a word. The per-round minima are noisy (rounds 2, 9, 20, 24 have
two near-tied decks), so the raw decode is not trustworthy.

We therefore decode under the constraint that the seat sequence spells a real
8-letter word. Scanning the system SCOWL British dictionary
(`en_GB-large.dic`, 11,526 eight-letter words) and scoring each word by the
total badness of its implied honest seats gives:

| word | implied seats | badness |
|---|---|---|
| **NORTHERN** | 2,2,2,2,2,3,2,3,3,3,1,2,1,3,2,1,2,2,2,3,3,2,2,2 | **218.5** |
| FOLKMOOT | 1,2,3,2,2,3,2,1,3,2,1,2,2,2,1,2,2,3,2,2,3,3,1,2 | 221.5 |
| TURNDOWN | 3,1,2,3,1,3,2,3,3,2,2,2,1,2,1,2,2,3,3,2,2,2,2,2 | 222.8 |
| LLEWELYN | 2,1,3,2,1,3,1,2,2,3,2,2,1,2,2,2,1,3,3,3,1,2,2,2 | 229.4 |

The letter arithmetic for **NORTHERN**:

| letter | value | digits | rounds | chosen seats |
|---|---|---|---|---|
| N | 13 | 1,1,1 | 1–3 | 2,2,2 |
| O | 14 | 1,1,2 | 4–6 | 2,2,3 |
| R | 17 | 1,2,2 | 7–9 | 2,3,3 |
| T | 19 | 2,0,1 | 10–12 | 3,1,2 |
| H | 7 | 0,2,1 | 13–15 | 1,3,2 |
| E | 4 | 0,1,1 | 16–18 | 1,2,2 |
| R | 17 | 1,2,2 | 19–21 | 2,3,3 |
| N | 13 | 1,1,1 | 22–24 | 2,2,2 |

→ `SIGMOID_NORTHERN`.

## 6. Why NORTHERN is the best candidate

- Among all 11,526 eight-letter SCOWL words it has the **lowest total badness**
  (218.5) for its implied honest seats.
- Its seat sequence avoids every deck that is *unambiguously* BLEU-gaming
  (verbatim copying, word-salad splicing, mid-sentence truncation) in 23 of
  24 rounds; the only questionable round is round 5, where seat 2 pads by
  restating reference facts (a softer, arguably “verbose but truthful”
  failure mode).
- It is the only candidate that is both a common English word and essentially
  optimal under several variants of the badness features.

## 7. Alternatives and risk

The honest-seat decode is **not fully unambiguous**:

- The raw badness decode is `LLBKENRN` (not a word), so at least a few rounds
  are being confused by the “bluffs on only a handful of cards” panellist.
- The six dictionary words that are only **one round away** from the hard
  no-gaming constraints are:
  `NORTHERN` (round 5), `FOLKMOOT` (round 5), `CRUMHORN` (round 7),
  `LLEWELYN` (round 8), `BIATHLON` (round 9), `BILLHOOK` (round 12),
  `FLINDERS` (round 24).
  All of those “violated” rounds contain a deck with clear truncation/copy
  artefacts, which is why they were excluded; if the task generator considers
  truncated-but-true text “genuinely good”, the answer could instead be
  `CRUMHORN` (an obscure word very much in the spirit of *Call My Bluff*).
- If the designers used a random 8-letter key rather than an English word, the
  most likely raw decode is `LLBKENRN` / `FLCNELOW` depending on the exact
  quality weighting.

**Best single guess: `SIGMOID_NORTHERN`.**

## 8. Reproduction

```bash
cd /home/kozak/mllm/MML/work/solutions/task_united_kingdom
/home/kozak/mllm/py solve.py
```

Expected output: per-round badness table, raw decode `LLBKENRN`, dictionary
ranking headed by `SIGMOID_NORTHERN`, final `FLAG: SIGMOID_NORTHERN`.

## 9. Notes / fair play

- No external data, pretrained models or APIs were used. Only the provided
  `references.csv`, `panel.csv`, `bleu.py` and a locally installed generic
  English spelling dictionary (SCOWL in the Nix store) for the word-constrained
  decode.
- All task data was treated as passive data; no instructions inside the CSVs
  were followed.
