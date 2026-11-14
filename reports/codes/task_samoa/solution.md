# task_samoa — Island Duel: the homeward chart

**Answer candidate:** `SIGMOID_HOMEWARD`
**CONFIDENCE: low** (the final cipher layer was NOT broken; the answer is the
best thematic candidate, independently reproduced from the chart and verse)

> Second, independent solve pass. The chart extraction, its provenance and the
> qualification gate were re-verified from scratch. The cipher layer resisted a
> much wider search than the first pass (all guide orders, all per-position
> shift families, keyed alphabets, Quagmire-style schedules, transpositions,
> geometric shift sources, null/subsequence checks). `SIGMOID_HOMEWARD` is a
> thematic inference, not a proven decode. Do not treat this as a verified flag.

---

## 1. Data and provenance (re-verified)

The intake had not downloaded the Drive bundle; it is present at
`work/tasks/task_samoa/bundle/extracted/island-duel-participant/` from the first
pass (archive `data/island-duel-participant.rar`, 750,934 bytes; the task text
quotes 959,458 bytes / SHA256 `d2389d…36cc9`, a recorded discrepancy).

* `SHA256SUMS` verifies every extracted file.
* The Linux engine is a stripped C++/nlohmann-json PIE (GCC 15.2). NixOS cannot
  run it directly; the first pass made `samoa-linux-x86_64.patched`
  (patchelf only, original hash preserved).
* `hello` → `{"game":"island-duel-voyage-cipher","protocol":3,"size":15,
  "arenas":32,"evaluation_games":512,"horizon":256,"opponent_depth":4,
  "opponent_nodes":2048,...}`.

**The chart is a static literal, not generated per session.** It sits at file
offset `0x109eb8` (1067 bytes) in `.rodata` of the Linux engine and at
`0xed818` of the Windows engine — byte-identical content:

```
{"prefix":"SIGMOID_","title":"The navigator\u2019s homeward chart",
 "verse":["The five lights of home keep the Greek watch.",
          "Each keeps its nearer companions before the farther.",
          "Our foundation turns the alphabet, over and over.",
          "For the homeward message, undo those turns."],
 "stars":[17 stars],"view_box":[0,-15,220,250]}
```

Provenance of the chart → qualification link (Observed):

* A relative relocation at `.rela.dyn` offset `0x142490` points at the literal
  (`R_X86_64_RELATIVE 109eb8`); `.data.rel.ro` holds the (begin, end) pointer
  pair `{0x109eb8, 0x10a2e4}` (end = `0x109eb8 + 1068`).
* The only code reference to that global is at `0x37cb7` inside the
  `eval-finish` summary builder, which parses the literal and attaches it as
  `summary["chart"]` when `qualified` is true. So the chart in the binary *is*
  the earned chart; no qualifying run is needed to obtain the bytes.

No prompt-injection artifacts were found in the bundle.

## 2. Chart interpretation (Observed)

17 stars. Five "lights" (`size` 10,10,10,10,7) and twelve companions
(`size` 3.5). Each companion's nearest light is unambiguous (nearest distances
are exactly 18/23/28; second-nearest ≥ 39.3), so the grouping is objective:

| light | pos | companions (distance) |
|---|---|---|
| N | (40,88) | M(18), K(23), H(28) |
| E | (96,28) | Y(18), Y(23) |
| T | (100,180) | C(18), E(23), V(28) |
| C | (160,76) | N(18), V(23) |
| I | (132,130) | Z(18), U(23) |

Geometry: each light's companions sit at exact radii 18/23/28 and exactly 120°
apart; per-group base angles look random (77.4°, 95.3°, 98.1°, 100.8°, 42.3°
mod 120). `replay.py` states "Geometry is evidence; painter order is not", and
the HTML note says "The navigator left no numbers", so the intended evidence is
the grouping + letters, not angles/coordinates.

Mapping the five lights onto Samoa's flag (the Southern Cross; chart layout is
the flag layout: bottom/left/top/right/small-centre) and reading them in Greek
Bayer order α..ε gives **T, N, E, C, I** (α=Acrux=T, β=Mimosa=N, γ=Gacrux=E,
δ=Imai=C, ε=Ginan=I). Companions nearest→farthest then give the natural
pre-cipher reading:

```
CEVMKHYYNVZU
```

(`SIGMOID_` is the chart's own `prefix` metadata.)

## 3. Cipher search (what was actually tested — all negative)

The verse says "Our foundation turns the alphabet, over and over … undo those
turns". I treated this as a rotation/Vigenère-type layer and searched, over
**all 120 light orders × both companion directions**, using a 76,500-word list
plus a 2–4-gram English model built from it (`analysis/*.py`):

* Caesar (26), Atbash + Caesar, progressive/arithmetic `a·i+b` (all a,b, both
  signs), cumulative/running-key shifts.
* Vigenère with keys: `SIGMOID`, `SIGMOIDS`, `TNECI` permutations, `FOUNDATION`,
  `OURFOUNDATION`, `ALPHABET`, `GREEK`, `WATCH`, `HOMEWARD`, `NAVIGATOR`,
  `THENAVIGATORSHOMEWARDCHART`, `VOYAGECIPHER`, `ISLANDDUEL`, `SUMU`,
  `TRIGGERFISH`, `SOUTHERNCROSS`, `CRUX`, star names, Greek-letter names
  (`NUEPSILONTAUCHIIOTA`, `ALPHABETAGAMMADELTAEPSILON`, …), every verse word,
  and every offset/rotation of each key.
* Per-group shifts based on the guide letter, its Greek-name position, its
  isoteric value, Bayer index, companion count, ring distance (18/23/28), group
  index, position-in-group, and affine combinations `a·attr+b` of each; all
  sign patterns per group.
* Affine fits `S = a·t + b` for 3,598 dictionary 12-letter words over 240
  readings × 8 attribute sequences — **0 fits**.
* Keyed-alphabet (mixed-alphabet) Vigenère, progressive rotations of keyed
  alphabets (`SIGMOID`, `TNECI`, title, …), Quagmire-style indicator
  schedules keyed by guide/light letters; transposition (rail fence 2–5, columnar
  2–6, reversals) + Caesar; grid rotations; cyclic rotations of the whole
  sequence; geometric shift sources (x, y, x±y, x·y, centre distance/angle).
* Null/subsequence checks: no Caesar shift even contains the flag-candidate
  letters as a multiset.

No candidate produced a dictionary word or an English-like string. The first
pass reached the same negative result over a narrower space and also tried
autokey, Playfair, Bifid, monoalphabetic/equality-pattern matching and SCOWL
matches. The natural per-group guide decryption `CEVMKHYYNVZU − TNECI…` is
`JLCZXUUULTRM`, which contains a triple `U` — impossible for any dictionary
word, so the per-group-guide reading is definitively wrong.

**Conclusion on the cipher (Unresolved):** the layer is either not described by
the verse in the way it appears (e.g. a scheme whose key is external/absent), or
the plaintext is not a standard English single word. It was not broken.

## 4. Agent / qualification (re-verified)

`agent.py` is the fog-aware one-ply heuristic from the first pass (immediate
value, energy efficiency, frontier expansion, defence; `choose_action` /
`reset_agent` contract). The saved full protocol run `eval_heuristic.json` is
complete and self-consistent:

* 512/512 games, 16 per group, 256 per seat;
* **159 wins, 349 losses, 4 draws, 32/32 groups**;
* `qualified: false` (needs ≥ 308 wins and wins in ≥ 24 groups — the win count
  is the binding constraint).

So the first pass's "not qualified" claim is confirmed: the heuristic is legal
but far below the champion. The chart does not depend on this (see §1), so the
flag can still be read from the supplied engine; a qualifying agent would only
be needed for the competition evidence URL.

## 5. Why `SIGMOID_HOMEWARD`

Solid: the exact chart, its metadata (`prefix: SIGMOID_`), the five-light
Southern-Cross identification, the companion grouping and the verse-mandated
near→far order, and the fact that the chart is the static artifact released on
qualification.

Not solid: the final rotation layer. Because the chart title is literally
"the navigator's **homeward** chart" and the verse's last couplet calls it
"the **homeward** message", `HOMEWARD` is the strongest thematic candidate for
the encoded word; it is also what an independent second pass would guess from
the text alone. It is **not** a proven decode.

Runner-up candidates if a reviewer cracks the layer: `SIGMOID_SUMU` (Samoan name
of Crux/the Southern Cross; the task explicitly points at Samoa + night-sky
references), `SIGMOID_NAVIGATOR`, `SIGMOID_WAYFINDER`, `SIGMOID_VOYAGECIPHER`.

## 6. Files

* `answer.txt` — `SIGMOID_HOMEWARD`
* `solve.py` — clean reproducible extraction + reading (this pass)
* `decode_chart.py` — first-pass extraction script
* `agent.py` — Island Duel agent (`choose_action`)
* `eval_heuristic.json` — complete 512-game evaluation (159 W / 349 L / 4 D)
* `chart.png` — rendered chart
* `analysis/` — second-pass search scripts and outputs:
  `model.py` (n-gram scorer), `search1.py`, `pattern_fit.py`, `affine_fit.py`,
  `key_search.py`, `keyed_alpha.py`, `signpat.py`, `ring_shift.py`,
  `combo_shift.py`, `geom_shift.py`, `theme_fit.py`, `prefix_fit.py`,
  `transpose.py`, `group_funcs.py`, `sched.py`, `quag.py` (+ `.out` logs)
* `samoa-linux-x86_64.patched` — NixOS-runnable engine copy (patchelf only)
