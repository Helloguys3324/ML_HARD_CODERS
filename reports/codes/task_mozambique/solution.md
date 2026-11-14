# The Professor of Maputo Bay (`task_mozambique`) — solution report

**Answer type:** `file` — submission is `main.py` (no text flag is derivable
locally; the evaluator releases `SIGMOID_<WORD>` only after a qualifying
submission). `answer.txt` contains the pointer `FILE_SUBMISSION:main.py`.

**CONFIDENCE: low** (see local scores below — the agent is legal, fast and
self-contained, but the local worst-of-5 match score is far below the 65%
threshold with the current heuristic).

## What was delivered
- `main.py` — self-contained submission (17 KB, stdlib only, PyPy-3.10 safe:
  no 3.11+ syntax, no third-party imports). Keeps the required `serve()` loop.
  Strategy in `agent(observation, memory)`.
- `solve.py` — reproducible local evaluation (imports `main.agent` and drives
  the official `arena.py` / `mozambique-engine`).
- `harness.py` — in-process Arena wrapper used for development.
- `agent_dev.py`, `agent_v3.py` — development snapshots (v1/v3 strategies).
- `eval.py` — parallel worst-of-5 map evaluator.

## How to reproduce
```bash
cd /home/kozak/mllm/MML
/home/kozak/mllm/py tools/fetch_drive_direct.py --task-id task_mozambique
cd work/tasks/task_mozambique/data
/home/kozak/mllm/py -c "import zipfile; zipfile.ZipFile('mozambique-participant.zip').extractall('.')"
chmod +x mozambique-engine
# NixOS: patch the ELF interpreter (the binary is built for generic glibc)
nix-shell -p patchelf --run 'patchelf --set-interpreter /nix/store/dphsh3jff8w6n9p3xbfmkia5krqwv52f-glibc-2.42-103/lib/ld-linux-x86-64.so.2 --set-rpath /nix/store/dphsh3jff8w6n9p3xbfmkia5krqwv52f-glibc-2.42-103/lib mozambique-engine'
# quick check of the protocol, exactly like the evaluator:
/home/kozak/mllm/py arena.py ../../solutions/task_mozambique/main.py --maps 2
# full local evaluation:
cd ../../solutions/task_mozambique && /home/kozak/mllm/py solve.py 24 1 4
```

## Engine facts established by experimentation (Observed)
- One action per round, 80 rounds; an action can only involve a boat that is
  not in transit. Sailing one route costs `routes[i][j]` rounds; a boat sails
  only to an adjacent harbour (route matrix non-zero), so pathfinding needs a
  next-hop table (implemented in `prep`).
- `load`/`unload` move goods between the boat and the *warehouse at its current
  site*; `gather` takes from natural stock. Capacity 3 goods per boat.
- `build_workshop` and `craft` are per-boat actions at a role-2 site; each
  consumes 2 timber + 1 fibre from that site's warehouse; craft output (1 good
  of type 2) appears in the site warehouse 3 rounds after the `craft` action.
- `deliver` tags one contract (`claims[i]` becomes the seat index of the
  claimant, `-1` = unclaimed) and consumes only the contract's goods; contracts
  are worth 5 (raw, qty 2) and 12 (crafted, qty 1). Deadlines are enforced by
  expiry (contracts left unclaimed after the deadline stay `-1`, verified for
  deadline-24 raw contracts).
- Natural stocks start at 4, cap 6, regenerate +1 every 4 rounds.
- Home warehouse starts at 5 timber + 3 fibre; a third boat costs 4 timber +
  2 fibre at home.
- Professor score on the same map varies with the game index (different
  deterministic styles) and adapts to our claims; in wait-only runs he scores
  39–68 on practice maps.

## Strategy implemented in `main.py`
1. Floyd–Warshall over the route graph + next-hop table.
2. Choose the manufacturing site (role 2) minimising
   `dist(home,F) + 0.7*(nearest timber + nearest fibre) + 0.35*(two nearest
   crafted destinations)`.
3. Opening: both boats load 2 timber + 1 fibre at home (4 actions), sail to F,
   unload, build the workshop, craft the first batch.
4. Sustained loop: keep the factory stocked (gather up to 3 timber / 2 fibre at
   the nearest sources), craft whenever the warehouse has 2T+1F and a boat is
   at F, load crafted goods and deliver to the earliest-deadline reachable
   unclaimed crafted contract (deadline feasibility is checked with travel
   time).
5. Opportunistic raw-contract missions (gather 2 units, sail, deliver) once the
   crafted pipeline is not starved; crafted cargo that has no target is
   unloaded back at the factory (deadlock fix).

## Local evidence (Observed)
- Protocol/time check: `arena.py main.py --maps 2` → `within_time_limit: true`,
  slowest game 0.14–0.35 s of Python thinking on CPython (PyPy is faster), so
  time-limits are safe.
- Full practice evaluations (`solve.py`, 16 maps, seed 1):
  `map_wins=0, map_draws=0, map_losses=16, match_score=0.0`.
  Typical game: us 24 (`2 crafted`), Professor 34–44 (`2–3 crafted + 2–3 raw`).
  On a few maps the agent reaches 29–36 (3 crafted), e.g. `[36, 32]` on
  group 0 / seed 767302775, which shows the win condition (3 crafted = 36 >
  max 32 for the Professor) is attainable, but not reliably.
- Key finding: total points on a map are fixed at 68 (4x5 raw + 4x12 crafted),
  so a win requires either (a) three crafted contracts (36 > 32) or (b) two
  crafted + at least three raw contracts (39 > 29). The current agent secures
  two crafted reliably but rarely competes for raw contracts, which is the main
  gap to 65%.
- The provided starter scores 0 on every practice game.

## Blockers / next steps
- The Professor's crafted rush is very fast (first crafted delivery round
  18–31 depending on map/game); a winning agent must deliver 2 crafted goods
  before ~round 35 and then deny him raw contracts, or produce 3 crafted
  before his second crafted claim.
- Best improvement path (not completed): a round-by-round scheduler that
  optimises both the factory logistics and early raw contracts jointly
  (e.g. beam search over task sequences using the deterministic rules above),
  plus explicit knowledge that `claims[i] == seat` (0 = us when we are seat 0,
  1 = us when we are seat 1).
- Nothing was submitted to the platform (per instructions).
