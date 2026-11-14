#!/usr/bin/env python3
"""main.py for LMML 2026 task_new_zealand — The Braided River Run.

Self-contained solver (standard library only). Reads ``test-input.txt``,
finds a minimum-time route via A* over the state (cell, tide-phase, bridge
state), and writes it to ``solution.txt``.

Action encoding: U/D/L/R move, W wait.  Standard library only, runs well under
the 2-minute limit.
"""
import heapq, time
from array import array
from collections import deque

WALL, WATER, FORD, SWAMP = "#", "=", "~", ","
MOVES = ["U", "D", "L", "R", "W"]
DELTAS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1), "W": (0, 0)}


def parse_map(text):
    header, winches = {}, {}
    grid, heights = [], []
    sec = "header"
    for raw in text.splitlines():
        if sec != "header":
            if not raw.strip():
                continue
            if raw.strip() == "heights" and sec == "map":
                sec = "heights"
                continue
            (grid if sec == "map" else heights).append(raw.rstrip("\n"))
            continue
        line = raw.split("#", 1)[0].strip() if not raw.lstrip().startswith("#") else ""
        if not line:
            continue
        p = line.split()
        k = p[0]
        if k == "map":
            sec = "map"
        elif k in ("tide_period", "tide_low", "swamp_cost", "climb_cost"):
            header[k] = int(p[1])
        elif k == "open":
            header["open"] = p[1:]
        elif k == "winch":
            winches[p[1]] = tuple(p[2:])
    H, W = len(grid), len(grid[0])
    hs = [[int(ch) for ch in row] for row in heights]
    start = goal = None
    for r in range(H):
        for c in range(W):
            if grid[r][c] == "S":
                start = (r, c)
            elif grid[r][c] == "G":
                goal = (r, c)
    return dict(P=header.get("tide_period", 10), K=header.get("tide_low", 3),
                swamp_cost=header.get("swamp_cost", 3), climb_cost=header.get("climb_cost", 1),
                winches=winches, open_bridges=set(header.get("open", [])), grid=grid,
                hs=hs, H=H, W=W, start=start, goal=goal)


def solve(text):
    m = parse_map(text)
    P, K = m["P"], m["K"]
    H, W = m["H"], m["W"]
    grid, hs = m["grid"], m["hs"]
    winches = m["winches"]

    bridges = sorted({ch for row in grid for ch in row if ch.isupper() and ch not in "SG"})
    bidx = {b: i for i, b in enumerate(bridges)}
    nb = len(bridges)
    nmask = 1 << nb
    winch_mask = {}
    for name, targets in winches.items():
        mm = 0
        for b in targets:
            mm |= 1 << bidx[b]
        winch_mask[name] = mm
    init_mask = 0
    for b in m["open_bridges"]:
        init_mask |= 1 << bidx[b]

    cell_id = [[-1] * W for _ in range(H)]
    cells = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] not in (WALL, WATER):
                cell_id[r][c] = len(cells)
                cells.append((r, c))
    ncell = len(cells)
    gr, gc = m["goal"]
    start_r, start_c = m["start"]

    # relaxed BFS (admissible lower bound on remaining time)
    h = array("i", [10 ** 9]) * ncell
    gid = cell_id[gr][gc]
    h[gid] = 0
    dq = deque([(gr, gc)])
    while dq:
        r, c = dq.popleft()
        d = h[cell_id[r][c]]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] not in (WALL, WATER):
                ci = cell_id[nr][nc]
                if h[ci] > d + 1:
                    h[ci] = d + 1
                    dq.append((nr, nc))

    sid = cell_id[start_r][start_c]
    goal_id = cell_id[gr][gc]

    def st_of(ci, tmod, mask):
        return (ci * P + tmod) * nmask + mask

    INF = 0x7FFFFFFF
    dist = array("i", [INF]) * (ncell * P * nmask)
    parent = array("i", [-1]) * (ncell * P * nmask)
    pact = array("B", [0]) * (ncell * P * nmask)

    start_state = st_of(sid, 0, init_mask)
    dist[start_state] = 0
    heap = [(h[sid], 0, start_state)]
    found = -1
    while heap:
        f, g, s = heapq.heappop(heap)
        if g > dist[s]:
            continue
        ci = s // (P * nmask)
        rem = s % (P * nmask)
        tmod = rem // nmask
        mask = rem % nmask
        if ci == goal_id:
            found = s
            break
        r, c = cells[ci]
        for a_i, a in enumerate(MOVES):
            dr, dc = DELTAS[a]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            sym = grid[nr][nc]
            if sym in (WALL, WATER):
                continue
            if a == "W":
                cost = 1
            else:
                cost = m["swamp_cost"] if sym == SWAMP else 1
                climb = hs[nr][nc] - hs[r][c]
                if climb > 0:
                    cost += m["climb_cost"] * climb
            ntmod = (g + cost) % P
            newmask = mask
            if sym == FORD:
                if ntmod >= K:
                    continue
            elif sym.isupper() and sym not in "SG":
                if not (mask >> bidx[sym]) & 1:
                    continue
            if a != "W" and sym in winch_mask:
                newmask ^= winch_mask[sym]
            ns = st_of(cell_id[nr][nc], ntmod, newmask)
            ng = g + cost
            if ng < dist[ns]:
                dist[ns] = ng
                parent[ns] = s
                pact[ns] = a_i
                heapq.heappush(heap, (ng + h[cell_id[nr][nc]], ng, ns))

    if found < 0:
        raise SystemExit("no route found")
    actions = []
    s = found
    while parent[s] >= 0:
        actions.append(MOVES[pact[s]])
        s = parent[s]
    actions.reverse()
    return "".join(actions)


def main():
    import os
    # the checker runs us in a folder that holds the map; allow either name
    path = "test-input.txt"
    if not os.path.isfile(path) and os.path.isfile("train-input.txt"):
        path = "train-input.txt"
    text = open(path, encoding="utf-8").read()
    route = solve(text)
    with open("solution.txt", "w", encoding="utf-8") as fh:
        fh.write(route)
    print("wrote solution.txt: %d actions" % len(route))


if __name__ == "__main__":
    main()