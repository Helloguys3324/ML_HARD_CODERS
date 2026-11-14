# Solution Report: task_australia ("The Drunken Mario Adventures")

## 1. Overview & Objective
- **Task ID:** `task_australia` (Australia).
- **Difficulty & Score:** Medium, 200 / 200 points.
- **Track:** Binary Reverse Engineering, Reinforcement Learning Forensics.
- **Context:** An experimental neuroevolution platform ("Marathon Lab") based on a 2D Mario-style platformer.
- **Artifact:** Standalone compiled ELF binary `marathon-lab-linux-x86_64` (27,990,042 bytes; SHA-256: `569dd52d9a4ff551798caff32c7eb3a1f5e462e14225d93f55a42dff25f2eca4`).
- **Anomaly:** The trained reference policy (`lab/proven.npy`) navigates the level while deliberately bypassing most visual coins and collecting only a specific subset, despite all coins appearing visually indistinguishable.
- **Objective:** Reverse-engineer environment mechanics, reward functions, and policy behavior to recover the secret flag.
- **Flag Format:** `SIGMOID_<WORD>`.

---

## 2. Key Insights & Methodology

1. **Nuitka Container Decompression:**
   Inspection of the binary footer revealed a Nuitka onefile bundle containing a zstd-compressed payload (signature `KA`). Unpacking extracted 448 application files, including the core ELF executable `marathon-lab.bin` and reference model weights `lab/proven.npy`.
2. **Runtime Introspection via `LD_PRELOAD`:**
   Because modules were compiled to native C, dynamic symbol hooking was implemented by injecting a C shim into the process using `LD_PRELOAD`. Calling the Python C API (`PyGILState_Ensure`, `PyRun_SimpleString`) exposed internal state across `lab.level` and `lab.fitness`.
3. **Reward Asymmetry Discovery:**
   The stage features 20 coins arranged along row $y=11$. The fitness evaluator `lab.fitness` maintains a hidden map `_SIGN`:
   - 7 coins have `_SIGN[(x, 11)] == 1`, awarding $+25.0$ reward;
   - 13 coins have `_SIGN[(x, 11)] == 0`, imposing a $-25.0$ penalty.
4. **Policy Verification:**
   Executing `lab/proven.npy` across 1,197 simulation steps (fitness score: 514.21) confirmed that the policy collects exactly the 7 positive coins `[(7,11), (17,11), (80,11), (101,11), (110,11), (118,11), (172,11)]`, skipping all 13 penalty coins.
5. **A1Z26 Cipher Decoding:**
   Arranging the 20 coin signs chronologically from left to right yields 4 five-bit integers:
   - Coins 1–5 ($x \in [7, 12, 17, 25, 32]$): `10100` $\to 20 \to$ **T**
   - Coins 6–10 ($x \in [43, 61, 66, 73, 80]$): `00001` $\to 1 \to$ **A**
   - Coins 11–15 ($x \in [93, 101, 110, 118, 130]$): `01110` $\to 14 \to$ **N**
   - Coins 16–20 ($x \in [157, 172, 195, 201, 206]$): `01000` $\to 8 \to$ **H**

The decoded token is **`TANH`**, yielding flag `SIGMOID_TANH`.

---

## 3. Step-by-Step Implementation Details

### Module Introspection Data
- **`lab.level` Constants:**
  - `WIDTH = 212`, `HEIGHT = 15`, `COIN_ROW = 11`.
  - `COIN_SITES = [7, 12, 17, 25, 32, 43, 61, 66, 73, 80, 93, 101, 110, 118, 130, 157, 172, 195, 201, 206]`.
- **`lab.fitness` Parameters:**
  - `_BASIS = 919247`, `PROGRESS = 1.0`, `TIME_COST = 0.01`, `COIN_WEIGHT = 25.0`, `WIN = 150.0`.
  - Hidden lookup `_SIGN` assigning binary signs $\{0, 1\}$ to each `(x, y)` coordinate.

### Verification Tools (`task_australia/`)
- `probe_runtime.py` / `probe_reward_events.py`: Dumps module tables and hooks step-level reward transitions.
- `solve.py`: Reconstructs the 20-bit array, verifies bit-chunking arithmetic, and outputs the decoded flag string.
- `evidence_*.txt`: Raw runtime logs validating the 1,197-step trajectory and coin pickup events.

---

## 4. Results & Artifacts
- **Status:** `PASSED` (Score: **200 / 200** points).
- **Submission Attempt:** Attempt 1 (`2026-11-14T12:27:02.112Z`, submitter: `Kozak`).
- **Recovered Flag:**
  ```text
  SIGMOID_TANH
  ```
- **Key Artifacts:**
  - `answer.txt` (`SIGMOID_TANH`);
  - `solution.md` (comprehensive forensic report);
  - `solve.py` (reproducible decoder script);
  - `coin_signs.csv` (coordinate-to-bit mapping table);
  - `evidence_module_dump.txt` and `evidence_reward_events.txt`.
