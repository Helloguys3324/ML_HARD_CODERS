#!/usr/bin/env python3
"""Router for "The Dispatch Desk of Fez" (task_morocco, LMML 2026).

Reads `test-input.csv` (columns: request_id,prompt) from the working folder and
writes `routing_log.csv` (columns: request_id,model) where `model` is either
`cheap` or `expensive`, one row per request.

How it works
------------
Both guides are real LLMs: the apprentice is a 0.5B model (`cheap`, 1 dirham),
the master a 7B model (`expensive`, 25 dirham).  We learn, from the 400 training
requests, which *kind* of question each guide answers correctly, then route new
requests by question kind.

Each request is given a "value" = estimated quality gain of using the master
over the apprentice (P(correct|expensive) - P(correct|cheap)), calibrated on the
training set:

   crates_remain        1.00   (cheap ~0%,  master ~100%)
   toll                 0.88   (cheap ~9%,  master ~97%)
   takings min/max      0.85   (cheap ~15%, master ~100%)
   time arithmetic      0.68   (cheap ~14%, master ~82%)
   board lookup(indir)  0.46   (cheap ~55%, master ~100%)
   takings-fee, "most"  0.45   (cheap ~23%, master ~69%)
   walk, direct=short   0.38   (cheap ~51%, master ~89%)
   crates_total         0.30   (cheap ~70%, master ~100%)
   takings-fee, "least" 0.12   (cheap ~27%, master ~40%)
   walk, indirect       0.07   (both poor)
   board lookup(direct) 0.00   (both always correct -> always cheap)

Requests are sorted by this value; the top `floor(19*N/24)` are routed to the
master, the rest to the apprentice.  The cost of `E` expensive + `C` cheap calls
is `25*E + C = N + 24*E`; requiring `E <= 19*N/24` keeps the total cost within
the 20-dirham-per-request budget exactly, for any number of requests.

Uses only the Python standard library and runs fully offline.
"""

from __future__ import annotations

import csv
import heapq
import re
from collections import defaultdict


def _walk_is_direct_shortest(prompt: str):
    """For a "shortest walk" question, return True iff the direct edge from the
    source to the target is the shortest path (Dijkstra on the quoted edges).

    The master guide answers the direct edge; she is reliable (~89%) only when
    that edge is actually the shortest, and both guides fail (~10% / ~3%) when a
    two-hop path is shorter.  Returns None if the prompt is not a walk question.
    """
    m = re.search(r"from\s+(\w+)\s+to\s+(\w+)", prompt)
    if not m:
        return None
    src, dst = m.group(1), m.group(2)

    edges = re.findall(r"([A-Za-z]+)-([A-Za-z]+)\s+(\d+)\s+minutes", prompt)
    graph = defaultdict(list)
    direct = None
    for a, b, w in edges:
        w = int(w)
        graph[a].append((b, w))
        graph[b].append((a, w))
        if (a == src and b == dst) or (a == dst and b == src):
            direct = w

    dist = {src: 0}
    heap = [(0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d != dist[u]:
            continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist.get(v, 10 ** 9):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))

    best = dist.get(dst)
    if direct is None or best is None:
        return False
    return direct == best


def routing_value(prompt: str) -> float:
    """Estimated quality gain of sending this request to the master guide."""
    if "When does" in prompt:
        return 0.68                                   # time arithmetic
    if "Which souk" in prompt:
        if "stall fee" in prompt:
            return 0.45 if "most" in prompt else 0.12  # takings minus fee
        return 0.85                                   # plain min/max takings
    if "shortest walk" in prompt:
        return 0.38 if _walk_is_direct_shortest(prompt) else 0.07
    if "What does" in prompt or "What do" in prompt:
        return 0.46 if "whatever" in prompt else 0.00  # indirect vs direct lookup
    if "How many" in prompt:
        return 1.00 if "remain" in prompt else 0.30   # multi-step vs single multiply
    if "total toll" in prompt:
        return 0.88
    return 0.30                                       # unseen template: lean safe


def main() -> None:
    with open("test-input.csv", newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    n = len(rows)
    expensive_limit = (19 * n) // 24   # cost = n + 24*E <= 20*n

    scored = [(r["request_id"], routing_value(r["prompt"])) for r in rows]
    scored.sort(key=lambda t: (-t[1], t[0]))   # highest value first, then by id
    expensive = {rid for rid, _ in scored[:expensive_limit]}

    with open("routing_log.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["request_id", "model"])
        for r in rows:
            model = "expensive" if r["request_id"] in expensive else "cheap"
            writer.writerow([r["request_id"], model])


if __name__ == "__main__":
    main()