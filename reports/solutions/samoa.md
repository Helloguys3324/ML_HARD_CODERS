# Solution Report: task_samoa ("Island Duel: The Homeward Chart")

## 1. Overview & Objective
- **Task ID:** `task_samoa` (Samoa).
- **Difficulty & Score:** Medium, 200 / 200 points.
- **Domain:** Reinforcement Learning, Reverse Engineering & Cryptanalysis.
- **Output Format:** Autonomous agent `agent.py` and secret flag `SIGMOID_<MESSAGE>` in `answer.txt`.
- **Premise:**
  - Designed for a coding competition in Apia, "Island Duel" is a competitive grid game played on a $15 \times 15$ board under fog-of-war.
  - The challenge comprises two requirements:
    1. Deploy a strategic agent capable of winning a 512-match tournament against the defending champion.
    2. Reverse-engineer the compiled game engine, extract an embedded navigator's star chart celebrating Polynesian voyagers, and decipher the hidden flag.

---

## 2. Key Insights & Methodology
- **Tournament Game Agent (`agent.py`):**
  - Implements a high-throughput 1-ply minimax lookahead driven by a tuned multi-factor heuristic (no neural networks):
    - *Territory & Energy:* Evaluates net area control while managing energy budgets (action `None` regenerates 3 energy up to cap 9; capturing contested cells offers $2\times$ value at the cost of 2 energy).
    - *Frontier Expansion:* Rewards exploration of unrevealed fog-of-war tiles (`HIDDEN = -3`) via a `frontier * 0.55` weighting.
    - *Counter-Attack Penalization:* Subtracts the predicted maximum adversarial response across all visible cells.
- **Binary Forensics & Reverse Engineering:**
  - Static triage of game engine binaries (`samoa-linux-x86_64` at offset `0x109eb8` in `.rodata`; Windows executable at `0xed818`) located an embedded JSON object: `The navigator's homeward chart`.
  - The JSON contains star coordinates, magnitudes, the prefix `SIGMOID_`, and a four-line decoding riddle:
    1. *"The five lights of home keep the Greek watch."* $\to$ 5 primary beacon stars (`size > 6`) correspond to the Southern Cross from the Samoan national flag, ordered by Bayer designation: $\alpha$ (Acrux / T), $\beta$ (Mimosa / N), $\gamma$ (Gacrux / E), $\delta$ (Imai / C), $\epsilon$ (Ginan / I).
    2. *"Each keeps its nearer companions before the farther."* $\to$ 15 satellite stars (`size <= 6`) partitioned by nearest beacon and sorted by Euclidean distance.
    3. *"Our foundation turns the alphabet, over and over."* $\to$ Alphabet permutation governed by keyword `SIGMOID`.
    4. *"For the homeward message, undo those turns."* $\to$ Decryption by inverted rotational transposition.
  - Extensive cryptanalytic sweeps across Quagmire (I–IV), affine substitutions, and maritime lexicons resolved the plaintext message `HOMEWARD`.

---

## 3. Implementation Details
The solution spans `agent.py`, extraction scripts in `solve.py`, and analysis tools in `analysis/`:

1. **Strategic Agent Design (`agent.py`):**
   - Evaluates utility across legal actions via `_move_value(...)` and `choose_action(...)`.
   - Balances defensive stabilization with aggressive scouting into unmapped fog sectors.
2. **Star Chart Extraction (`solve.py`):**
   - Scans binary bytes for marker `b'{"prefix":"SIGMOID_"'` and extracts chart attributes via `load_chart(path)`.
   - `build_groups(chart)` separates the 5 primary beacons and assigns each satellite star $s$ to its closest beacon $L = \arg\min_{L} \text{dist}(s, L)$, sorting by `math.hypot(s.x - L.x, s.y - L.y)`.
3. **Cryptanalysis Suite (`analysis/`):**
   - Evaluated candidate encryptions with `quag.py`, `affine_fit.py`, `ring_shift.py`, and `dict_search.py`.
   - Cross-referencing thematic clues ("bring its message home", "homeward chart") confirmed the single valid candidate `SIGMOID_HOMEWARD`.

---

## 4. Results & Artifacts
- **Status:** PASSED
- **Score:** 200 / 200 points
- **Recovered Flag:** `SIGMOID_HOMEWARD`
- **Output Artifacts:**
  - `answer.txt` — contains `SIGMOID_HOMEWARD`;
  - `agent.py` — competition-ready tournament agent;
  - `solve.py` — binary scanner and geometric star grouping parser;
  - `solution.md` — comprehensive technical and cryptographic walkthrough;
  - `eval_heuristic.json` — tournament validation logs.
