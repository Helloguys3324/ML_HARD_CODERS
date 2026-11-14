"""task_senegal — "The Teranga FM Token Diet" solver.

Finds the minimum-token routing config (tier, style, fewshot, context) per
question that still clears every question's quality bar, then verifies the
total against the 1200-token budget.

The entire cost/quality mechanism lives in `model_utils.py` (deterministic,
offline). Key facts derived by reading it:

- `render_completion_body(slots, tier, style, context)` includes
  `slot["phrasings"][0]` for a slot iff `assisted or not slot["large_only"]`,
  where `assisted = (tier == "large") or context`.
- So a question passes deterministically iff every slot gets included, i.e.
  iff `assisted` is True OR the question has no `large_only` slots.
- `context` only helps (a) questions that HAVE a context_snippet, and
  (b) cheap tier; on large tier a snippet is pure waste, and on a question
  with no snippet `context: true` is a no-op (build_call uses `context and
  bool(snippet)`).

Therefore the optimum is trivially: for every question use the CHEAPEST
(tier, style, fewshot, context) combination that still includes all slots.
We brute-force all 16 combinations per question (cheap borrows nowhere, ~0s).
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import model_utils as mu

HERE = Path(__file__).resolve().parent
FACT_BANK = mu.load_fact_bank(HERE / "dataset" / "fact_bank.json")

TIERS = ("cheap", "large")
STYLES = ("verbose", "terse")
BOOLS = (False, True)


def cost_quality_map():
    """For each question, tabulate (tokens, pass) for every combo."""
    rows = {}
    for qid, q in FACT_BANK.items():
        combos = []
        for tier, style, fewshot, context in itertools.product(TIERS, STYLES, BOOLS, BOOLS):
            call = mu.build_call(q, tier, style, fewshot, context)
            passed = mu.question_passes(q["slots"], call["answer"])
            combos.append({
                "tier": tier, "style": style, "fewshot": fewshot,
                "context": context, "tokens": call["tokens"], "pass": passed,
            })
        rows[qid] = combos
    return rows


def solve():
    rows = cost_quality_map()

    config = {}
    total = 0
    details = []
    for qid, combos in rows.items():
        passing = [c for c in combos if c["pass"]]
        assert passing, f"{qid}: no combo passes?!"
        best = min(passing, key=lambda c: c["tokens"])
        config[qid] = {k: best[k] for k in ("tier", "style", "fewshot", "context")}
        total += best["tokens"]
        details.append((qid, best))

    # Verbose accounting for the writeup.
    print("== per-question optimal routing ==")
    for qid, b in details:
        q = FACT_BANK[qid]
        n_large_only = sum(1 for s in q["slots"] if s["large_only"])
        has_ctx = bool(q.get("context_snippet"))
        print(f"{qid}: {b['tier']:5s} {b['style']:7s} fewshot={int(b['fewshot'])} "
              f"context={int(b['context'])}  tokens={b['tokens']:3d}  "
              f"large_only={n_large_only} has_ctx={int(has_ctx)}")
    print(f"\nTOTAL tokens = {total} (budget {mu.TOKEN_BUDGET}, cap delta {mu.TOKEN_BUDGET - total})")

    # Reproduce the full run through the exact gateway to double-check gates.
    result = mu.run_config(config, FACT_BANK)
    print("\n== end-to-end run_config check ==")
    print(f"quality_gate = {result['quality_gate']}  budget_gate = {result['budget_gate']}")
    print(f"total_tokens = {result['total_tokens']}  passes = "
          f"{sum(1 for i in result['per_question'].values() if i['pass'])}/20")
    assert result["quality_gate"] and result["budget_gate"], "gates failed!"
    assert result["total_tokens"] == total

    # Baseline (intern's config) for the writeup's "economy" figure.
    naive = {qid: {"tier": "large", "style": "verbose", "fewshot": True, "context": False}
             for qid in FACT_BANK}
    naive_result = mu.run_config(naive, FACT_BANK)
    print(f"\nnaive baseline tokens = {naive_result['total_tokens']} "
          f"(saved {naive_result['total_tokens'] - total} tokens, "
          f"{(naive_result['total_tokens'] - total) / naive_result['total_tokens']:.1%})")

    return config, total, naive_result["total_tokens"]


def write_outputs(config: dict):
    mu.save_json(config, HERE / "routing_config.json")
    mu.save_json(config, HERE / "submission.json")
    print(f"\nwrote routing_config.json and submission.json to {HERE}")


if __name__ == "__main__":
    config, total, naive_tokens = solve()
    write_outputs(config)
    print(f"\nFINAL: total={total}, saved={naive_tokens - total} tokens, "
          f"budget={mu.TOKEN_BUDGET}")