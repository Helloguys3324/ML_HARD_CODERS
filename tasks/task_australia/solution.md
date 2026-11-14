# Task australia — The Drunken Mario Adventures (Marathon Lab)

**Answer: `SIGMOID_TANH`**

**CONFIDENCE: high**

## TL;DR

Marathon Lab hides a 20-bit message in the **reward sign of the 20 coin sites**.
A merely competent player collects every coin; the fully trained agent learns
that most of the coins are traps. In its stable, clean run it collects exactly
the seven +25 coins and avoids all thirteen −25 coins. Reading the coin row
left to right as `+ = 1`, `− = 0` gives

```
10100 00001 01110 01000  ->  20 1 14 8  ->  T A N H
```

so the flag is **`SIGMOID_TANH`** (a fitting activation-function joke next to
the `SIGMOID_` prefix).

## What I did

1. Downloaded the Linux build from the Dropbox link in `task.json`
   (`work/tasks/task_australia/data/marathon-lab-linux-x86_64`,
   27,990,042 bytes,
   SHA-256 `569dd52d9a4ff551798caff32c7eb3a1f5e462e14225d93f55a42dff25f2eca4`).
2. Identified it as a **Nuitka one-file** application: the ELF ends with a
   `KA` + zstd payload; the last 8 bytes are the little-endian payload length.
   I decompressed the payload and extracted 448 files, including
   `marathon-lab.bin`, `lab/proven.npy`, Pygame and SDL.
3. Ran the app headlessly under Xvfb inside a bubblewrap FHS (NixOS cannot run
   the generic glibc ELF directly). `./marathon-lab.bin --selfcheck` prints
   `OK: linux, data files bundled, one generation ran.`
4. Because the game logic is compiled by Nuitka (no `.pyc`), I injected a small
   `LD_PRELOAD` library that calls the exported `PyGILState_Ensure` and
   `PyRun_SimpleString` after startup. That lets ordinary Python run inside the
   game process. Through it I dumped
   `lab.level.COIN_SITES`, `lab.fitness._SIGN` and the `Lab`/`Brain` APIs
   (`probe_runtime.py`, `inject.c`, `evidence_module_dump.txt`).
5. Ran the real `Lab` evolution (random population, no external data) for 15
   generations to confirm the learning trend, and evaluated the bundled,
   fully-trained reference brain `lab/proven.npy` deterministically.

No external datasets, pretrained models or APIs were used. Everything came
from the supplied executable.

## Evidence

### The level has 20 coins in one row

From the live module dump (`evidence_module_dump.txt`):

```
WIDTH=212 HEIGHT=15 COIN_ROW=11
COIN_SITES = [7, 12, 17, 25, 32, 43, 61, 66, 73, 80,
              93, 101, 110, 118, 130, 157, 172, 195, 201, 206]
```

### `lab.fitness` marks each coin as a reward or a trap

```
_BASIS = 919247
PROGRESS = 1.0
TIME_COST = 0.01
COIN_WEIGHT = 25.0
DEATH = 0.0
WIN = 150.0
_SIGN = {
  (7,11):1, (12,11):0, (17,11):1, (25,11):0, (32,11):0,
  (43,11):0, (61,11):0, (66,11):0, (73,11):0, (80,11):1,
  (93,11):0, (101,11):1, (110,11):1, (118,11):1, (130,11):0,
  (157,11):0, (172,11):1, (195,11):0, (201,11):0, (206,11):0
}
```

Observed per-coin reward in an actual episode: `+25` for a `_SIGN`=1 coin,
`−25` for a `_SIGN`=0 coin (plus a tiny progress/time term). This is exactly
the story's "goes out of his way for things he should walk past, and walks past
things he should want": all coins look identical, but the trained agent knows
which ones are traps.

### The fully trained agent takes exactly the +1 coins

Running the bundled trained genome `lab/proven.npy` on three consecutive clean
runs gives an identical, fitness-stable episode:

```
proven run 0: steps=1197 fitness=514.21 won=True max_x=204.18
  coins=[(7,11),(17,11),(80,11),(101,11),(110,11),(118,11),(172,11)]
proven run 1: identical
proven run 2: identical
```

Those are **exactly the seven `_SIGN`=1 sites**, and none of the thirteen
`_SIGN`=0 sites. Across further runs the agent consistently takes all seven
`+1` coins; occasionally it also clips one `-1` coin in passing (e.g. `(12,11)`,
which scores `-24.825`), while still completing the level. The selective
behaviour is unmistakable: it seeks out the `+25` coins and walks past almost
all `-25` traps. A per-coin reward trace is in
`evidence_reward_events.txt`:

```
(7,11)   +25.012   -> total 28.354
(12,11)  -24.825   -> total  7.444   (occasional incidental -1 coin)
(17,11)  +25.144   -> total 36.914
(80,11)  +25.175   -> ...
(101,11) +25.175
(110,11) +25.175
(118,11) +25.175
(172,11) +25.175
```

The evolution run (24 agents, 15 generations, starting from random genomes;
0 exact copies of the proven brain in the initial population) showed the same
direction: the best agents started grabbing coin 7, then 17, with leader score
rising 20.6 → 86.8; the trained reference brain completes the full pattern.

### Decoding

```
site x :   7 12 17 25 32 43 61 66 73 80 93 101 110 118 130 157 172 195 201 206
sign   :   1  0  1  0  0  0  0  0  0  1  0   1   1   1   0   0   1   0   0   0
bitstr :   10100 00001 01110 01000
A1Z26  :    20    1    14     8
letter :     T    A     N     H
```

Equivalently, if we ignore the source and just mark the three clean trained-agent
runs' collected sites as 1 and the untouched sites as 0, we get the same
`10100000010111001000`, i.e. the message is readable directly from the agent's
choices. All four groups are valid A–Z values. The inverse/other groupings do
not yield valid letters or a word; `TANH` is also semantically perfect for a
task whose flag prefix is `SIGMOID_`.

## Reproduction

```bash
# 1. download
curl -L -o data/marathon-lab-linux-x86_64 \
  'https://dropbox.com/scl/fi/6vzyji3zk8le8i95yc35r/marathon-lab-linux-x86_64?dl=1&rlkey=ssw73nzuor6qbjiy8bscy23jm&st=17adwv90'
chmod +x data/marathon-lab-linux-x86_64

# 2. decode the message from the recovered reward mask
/home/kozak/mllm/py work/solutions/task_australia/solve.py
# -> decoded word: TANH ; flag: SIGMOID_TANH

# Optional: re-run the compiled game's own self-check
#   (needs a system Python/glibc FHS; on NixOS I used the bubblewrap wrapper
#    run_fhs.sh, or `nix-shell -p steam-run` from the extracted dist)
./marathon-lab.bin --selfcheck
# -> OK: linux, data files bundled, one generation ran.
```

`probe_runtime.py` + `inject.c` contain the live-introspection probe used to
dump `_SIGN`, `COIN_SITES` and the `proven.npy` behaviour (run the app with
`LD_PRELOAD=inject.so`; the probe writes `/tmp/probe_out.txt` and
`/tmp/dump.txt`). `evidence_runtime.txt` is the raw output.

## Files

- `answer.txt` — `SIGMOID_TANH`
- `solve.py` — decoder / verifier (parses the captured module dump and the
  trained-agent coin list, prints the flag)
- `solution.md` — this file
- `evidence_module_dump.txt` — live `lab.*` module globals dumped from the app
- `evidence_runtime.txt` — raw trained-brain / evolution probe output
- `evidence_reward_events.txt` — per-coin +25/−25 reward trace
- `probe_runtime.py`, `probe_reward_events.py`, `inject.c`, `run_fhs.sh` —
  runtime introspection harness
- `coin_signs.csv` — per-coin site, sign, reward, group/bit
- `screenshot_gen0.png` — the running app at generation 0
