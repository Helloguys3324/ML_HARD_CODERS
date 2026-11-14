# Solution — `task_papua_new_guinea` (Bihainim Angkol / Follow Your Uncle)

**Task type:** `file` → submit `main.py` (200 pts, Medium)
**Deliverable:** `work/solutions/task_papua_new_guinea/main.py`
**Confidence: HIGH** — the approach is provably optimal for this MDP, and it was validated
end-to-end against the official simulator.

---

## 1. What the task gives us

`lagoon.py` is a *complete, readable simulator* of a 24-step day on a 7x6 grid:

| cell | where | bite chance per `FISH` |
|---|---|---|
| `s` seagrass flats | (2,4),(2,5),(3,4),(3,5) — inside the reef | 0.20 |
| `T` outer reef drop-off | (0,4),(0,5) — open sea, north of the reef | **0.90** |
| `H` beach | (5,2) — landing happens here | — |
| `~` reef passage | (1,4) — the only way through the barrier reef; ends a step there → pushed 1 cell south with p=0.40 | — |

Constants: `STEPS_PER_DAY=24`, `HOLD_CAPACITY=6`, `TIDE_CHANCE=0.40`.
Bite/tide draws are per-step-index, i.i.d. from the day's seed.

The submission is a **table** `(row,col,hold,steps_left) -> action`, i.e. a deterministic
stationary policy over a small finite MDP (36 water cells x 7 holds x 24 step counts =
**6048 states**). That is exactly the shape of a dynamic-programming solution.

## 2. Sanity check: the simulator reproduces angkol

I replayed angkol's logged 400 actions through the *unmodified* `lagoon.py`:

```
days=400  action-replay mismatches in final catch: 0
angkol mean (replay) = 2.490   logged = 2.490
```

So the rules are understood exactly and there is no hidden rule. Angkol's mean is
**2.49 fish/day** (distribution 0..6, sd 1.56). Target: **>= 1.5 x 2.49 = 3.735**.

Angkol only ever works the **seagrass** (20 % bites, ~11-13 `FISH` attempts), never
crosses the passage to the **trout drop-off** (90 % bites). That is the gap.

## 3. Method: exact backward induction (no learning needed)

Because the model is known and the policy format is a state table, the optimal
expected catch is computable in closed form with a Bellman backup:

```
V[k][(r,c,h)] = max_a  E_{bite,tide}[ landed + V[k-1][next_state] ]
V[0] = 0,   k = steps_left = 1..24
```

Transition (mirrors `lagoon._advance` exactly): apply the action, then the tide if the
canoe *ends* the step on `~`, then automatic landing if on `H` (hold -> score, hold=0).
`FISH` succeeds with p = BITE_CHANCE[cell] only when `hold < 6`.

One subtlety handled correctly: the bite roll uses the cell the canoe is on **before**
the move, and the tide roll is applied **after** the move — so `FISH` at a ground and
the tide never interact, while a *passage crossing* costs 1/0.6 - 1 = 0.67 extra steps
northbound and saves steps southbound (the push lands you where you were going anyway).

Result:

```
optimal expected catch from (beach, hold 0, 24 steps) = 5.9575
angkol = 2.490  ->  expected ratio 2.393x   (needed 1.5x)
```

`5.96/6.00` — this is essentially the **hard ceiling of the game** (hold capacity 6,
and only one full excursion to the drop-off fits inside 24 steps). No policy can do
meaningfully better.

## 4. The policy the DP found (interpretable)

```
(5,2) NORTH NORTH NORTH EAST EAST NORTH [passage] NORTH -> (0,4)
FISH x ~7   (90% bites -> 6 fish)
SOUTH [passage, 40% free ride] SOUTH SOUTH SOUTH SOUTH WEST WEST -> (5,2) land 6
```

It leaves the beach, runs north along col 2, cuts east to the passage at (2,4),
crosses to the trout drop-off, fills the hold, and rides the tide back south.
Near the end of the day the DP automatically settles for a partial hold and lands it
(never leaves fish in the canoe).

Sample day (`--day 1`): 6 fish, landed at `steps_left=1`.
Simulated 20 000 unseen days (seeds 1-20000): **mean 5.959, sd 0.277, 97.3 % of days
at the 6-fish cap**, worst day 0 (1 in 20 000).

## 5. Validation

| check | result |
|---|---|
| Angkol replay through `lagoon.py` | 0/400 catch mismatches, mean 2.490 |
| DP policy on angkol's own 400 days | 5.963 fish/day → **2.395x** |
| DP policy on 2 000 fresh days (seeds 1001-3000) | 5.965 fish/day → **2.395x** |
| 20 000 fresh days | 5.959 fish/day, 97.3 % at cap |
| `lagoon.py --policy test-output.csv --compare train-output.json` | `ratio 2.395 x angkol (target 1.5 x): above the target` |
| Output format | 6048 rows, 1 per state, same order as input, no duplicates, all actions valid |
| Runtime of `main.py` | **0.6 s** (limit 60 s), stdlib only (`base64`, `csv`, `zlib`) |
| Runs alone next to `test-input.csv` only | yes (policy is a compressed table inside the file) |
| Shuffled `test-input.csv` | works (lookup by state, not by row position) |
| Unknown/extra state in input | falls back to `WAIT` (valid action, no crash) |

Robustness to the hidden day sample (bootstrap of angkol's 400 daily catches, threshold
angkol mean > 5.96/1.5 = 3.97 to fail):

| hidden days | P(ratio < 1.5) |
|---|---|
| 10 | 0.14 % |
| 20 | < 0.005 % |
| 50 / 100 / 400 | ~0 |

Our catch is near-constant at ~6, so risk only comes from an unrealistically lucky
angkol sample — and even his best day ever is 6.

## 6. Files

| file | contents |
|---|---|
| `main.py` | **the submission** — self-contained table policy for all 6048 states (20.9 KB) |
| `solve.py` | reproducible solver: replay check, DP, validation, writes `policy.csv` + `main.py` |
| `policy.csv` | human-readable optimal policy table |
| `solution.md` | this report |

`main.py` sha256: `a15f29f0e0df4be1f416d80aa9be44ac37a8dba5a00a31f9dc4445a6be3e60b7`

## 7. Commands

```bash
cd /home/kozak/mllm/MML/work/solutions/task_papua_new_guinea
/home/kozak/mllm/py solve.py            # ~11 s: replay, DP, hold-out, writes main.py

# grader replica
mkdir /tmp/opencode/angkol_check && cp main.py /tmp/opencode/angkol_check/
cp ../../tasks/task_papua_new_guinea/data/task_papua_new_guinea/train-input.csv \
   /tmp/opencode/angkol_check/test-input.csv
cp ../../tasks/task_papua_new_guinea/data/task_papua_new_guinea/{lagoon.py,train-output.json} \
   /tmp/opencode/angkol_check/
cd /tmp/opencode/angkol_check && /home/kozak/mllm/py main.py \
  && /home/kozak/mllm/py lagoon.py --policy test-output.csv --compare train-output.json
```

## 8. Notes / fair play

* No external data, pretrained models or APIs — only the provided simulator and logbook.
* The logbook was used as evidence (angkol's behaviour and his 2.49 baseline) and to
  validate the simulator; the policy itself comes from the published rules, which the
  task explicitly encourages ("`lagoon.py` holds every rule and constant above").
* Nothing was submitted to the platform. Hand-off to the orchestrator.
