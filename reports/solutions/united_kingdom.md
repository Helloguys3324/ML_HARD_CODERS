# Solution Report: task_united_kingdom ("Call My BLEUff")

## 1. Overview & Objective
- **Task ID:** `task_united_kingdom` (United Kingdom).
- **Difficulty:** Easy / Medium.
- **Domain:** NLG Metrics Gaming / NLP Forensics + Base-3 (Ternary) Steganographic Decoding.
- **Output Format:** Text string `SIGMOID_<WORD>` (8-letter English word).
- **Input Data:**
  - `references.csv`: 24 rounds $\times$ 40 restaurant cards $\times$ 5 reference descriptions = 4,800 rows.
  - `panel.csv`: 24 rounds $\times$ 40 cards $\times$ 3 panel seats (seat 1, seat 2, seat 3) = 2,880 descriptions. A "deck" is the set of 40 cards submitted by one seat in one round.
  - `bleu.py`: Unsmoothed corpus BLEU-4 in lowercase with brevity penalty against the closest reference length.
- **Premise & The BLEU Paradox:**
  - Styled after the classic UK television show *Call My Bluff*. In each round, three panelists propose descriptions: two bluffers game the BLEU metric, while one is an honest narrator.
  - Selecting the highest-BLEU deck in each round yields the ungrammatical gibberish `SKELDRAP`. Because standard BLEU measures only n-gram precision, it heavily rewards adversarial truncation, padding, splicing, and verbatim copying.
- **Goal:** Develop an anti-gaming "dishonesty" penalty metric to identify the true honest seat across all 24 rounds and decode the underlying 8-letter English word.

---

## 2. Key Insights & Methodology

### 2.1 Anatomy of BLEU Gaming Tactics
Analysis of high-BLEU decks revealed four distinct adversarial manipulation patterns:
1. **Verbatim Plagiarism / Concatenation:** Merging two complete reference sentences ($p_4 \approx 1.0$), artificially yielding BLEU scores of 0.74–0.78 (up to 62 copied cards per deck).
2. **Padding & Fact Duplication:** Redundantly repeating facts to inflate n-gram overlap and counter the brevity penalty.
3. **Aggressive Truncation:** Cutting sentences mid-clause ("It is a kids.", "This pub."), maximizing short n-gram precision while absorbing mild corpus-level brevity penalties.
4. **Splicing:** Stitching fragmented clauses into ungrammatical franken-sentences ("A high end family it a 3 out of 5 customer rating...").

### 2.2 Listener-Side "Dishonesty" Penalty Metric
For each of the 2,880 generated descriptions, penalties were computed across indicators ignored by BLEU:
- **Verbatim Copy:** $p_4 \ge 0.8$ or `difflib.SequenceMatcher` similarity $\ge 0.95$.
- **Plagiarism:** $\ge 85\%$ of 5-grams present in reference sentences.
- **Length Distortion:** Length ratio to nearest reference $< 0.72$ (truncation) or $> 1.30$ (padding).
- **Omission / Low Recall:** Information recall $< 0.55$ across 5 slot types (price, rating, family-friendliness, area, cuisine).
- **Dangling Truncation:** Trailing prepositions or conjunctions (`with`, `in`, `to`, `at`).
- **Hallucination:** Direct contradiction of reference facts.
A deck's total badness is the sum of penalties across its 40 cards; the honest panelist minimizes this score.

### 2.3 Base-3 (Ternary) Word Decoding
- The 24 rounds correspond to 8 three-round triplets (rounds 1–3, 4–6, ..., 22–24), each encoding one letter.
- In each round, the honest seat $s \in \{1, 2, 3\}$ represents ternary digit $d = s - 1 \in \{0, 1, 2\}$.
- The letter value is given by:
  $$\text{value} = d_0 \cdot 9 + d_1 \cdot 3 + d_2 \quad (0 \to \text{'A'}, \dots, 25 \to \text{'Z'})$$
- Raw greedy minimum penalties yielded the noisy string `LLBKENRN`.
- **SCOWL Dictionary Vetting:** Constraining the sequence against all 11,526 eight-letter words in the British SCOWL dictionary (`en_GB-large.dic`) identified **`NORTHERN`** as the global minimum penalty candidate (score: **218.5**):
  - **N** (13) $\to (1, 1, 1) \to$ Rounds 1–3: Seats (2, 2, 2)
  - **O** (14) $\to (1, 1, 2) \to$ Rounds 4–6: Seats (2, 2, 3)
  - **R** (17) $\to (1, 2, 2) \to$ Rounds 7–9: Seats (2, 3, 3)
  - **T** (19) $\to (2, 0, 1) \to$ Rounds 10–12: Seats (3, 1, 2)
  - **H** (7)  $\to (0, 2, 1) \to$ Rounds 13–15: Seats (1, 3, 2)
  - **E** (4)  $\to (0, 1, 1) \to$ Rounds 16–18: Seats (1, 2, 2)
  - **R** (17) $\to (1, 2, 2) \to$ Rounds 19–21: Seats (2, 3, 3)
  - **N** (13) $\to (1, 1, 1) \to$ Rounds 22–24: Seats (2, 2, 2)
- Runners-up: `FOLKMOOT` (221.5), `TURNDOWN` (222.8), `LLEWELYN` (229.4).

---

## 3. Implementation Details

1. **`solve.py` (Reproducible Forensics & Decoding Pipeline):**
   - Parses `references.csv` and `panel.csv`.
   - Computes baseline BLEU-4 per deck, demonstrating the decoy word `SKELDRAP`.
   - Evaluates the card-level badness matrix ($24 \times 3$).
   - Scores candidate dictionary words via ternary seat mappings and outputs `SIGMOID_NORTHERN`.

2. **`solution.md`:**
   - Detailed round-by-round breakdown of seat penalties and specific gaming tactics identified.
   - Mathematical proof of the ternary encoding scheme and dictionary ranking.

---

## 4. Results & Artifacts
- **Flag Candidate:** `SIGMOID_NORTHERN`
- **Confidence:** Medium / Heuristic (top match among 11,526 British dictionary entries with minimum total penalty, strictly avoiding BLEU gaming artifacts).
- **Key Artifacts (`task_united_kingdom/`):**
  - `answer.txt`: Contains `SIGMOID_NORTHERN`.
  - `solve.py`: Automated BLEU audit, penalty scoring, and dictionary decoding script.
  - `solution.md`: Technical report detailing gaming mechanisms, round penalty matrices, and candidate comparisons.
