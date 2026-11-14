#!/usr/bin/env python3
"""Solver for LMML 2026 task_new_zealand — The Braided River Run.

Finds a minimum-time route from S to G via A* over the expanded state space
(position, tide-phase, bridge state). Deterministic; standard library only.
"""
from __future__ import annotations
import sys, os, heapq, time
from array import array

U, D, L, R, W = "U", "D", "L", "R", "W"
DELTAS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1), "W": (0, 0)}
MOVES = list(DELTAS.keys())
WALL, WATER, FORD, SWAMP = "#", "=", "~", ","


def parse_map(text: str):
    header = {}
    winches = {}
    grid = []
    heights = []
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
    P = header.get("tide_period", 10)
    K = header.get("tide_low", 3)
    swamp_cost = header.get("swamp_cost", 3)
    climb_cost = header.get("climb_cost", 1)
    open_bridges = set(header.get("open", []))
    H = len(grid)
    Wd = len(grid[0])
    hs = [[int(ch) for ch in row] for row in heights]
    start = goal = None
    for r in range(H):
        for c in range(Wd):
            if grid[r][c] == "S":
                start = (r, c)
            elif grid[r][c] == "G":
                goal = (r, c)
    return dict(P=P, K=K, swamp_cost=swamp_cost, climb_cost=climb_cost,
                winches=winches, open_bridges=open_bridges, grid=grid,
                hs=hs, H=H, W=Wd, start=start, goal=goal)


def solve(text: str, budget=None):
    st = time.time()
    m = parse_map(text)
    P, K = m["P"], m["K"]
    H, Wd = m["H"], m["W"]
    grid, hs = m["grid"], m["hs"]
    winches = m["winches"]

    # bridge bit assignment
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

    # walkable cell indexing (exclude wall & deep water)
    cell_id = [[-1] * Wd for _ in range(H)]
    cells = []
    for r in range(H):
        for c in range(Wd):
            if grid[r][c] not in (WALL, WATER):
                cell_id[r][c] = len(cells)
                cells.append((r, c))
    ncell = len(cells)
    gr, gc = m["goal"]
    start_r, start_c = m["start"]

    # relaxed BFS distance to goal (each step cost 1, ignore tide/bridge/swamp/climb)
    h = array("i", [10**9]) * ncell
    gid = cell_id[gr][gc]
    h[gid] = 0
    from collections import deque
    dq = deque([(gr, gc)])
    while dq:
        r, c = dq.popleft()
        d = h[cell_id[r][c]]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < Wd and grid[nr][nc] not in (WALL, WATER):
                ci = cell_id[nr][nc]
                if h[ci] > d + 1:
                    h[ci] = d + 1
                    dq.append((nr, nc))

    sid = cell_id[start_r][start_c]
    goal_id = cell_id[gr][gc]

    def st_of(ci, tmod, mask):
        return (ci * P + tmod) * nmask + mask

    total = ncell * P * nmask
    INF = 0x7FFFFFFF
    dist = array("i", [INF]) * total
    parent = array("i", [-1]) * total
    pact = array("B", [0]) * total

    start_state = st_of(sid, 0, init_mask)
    dist[start_state] = 0
    hval = h[sid]
    heap = [(hval, 0, start_state)]
    push = heapq.heappush
    pop = heapq.heappop
    found = -1
    expanded = 0
    while heap:
        f, g, s = pop(heap)
        if g > dist[s]:
            continue
        ci = s // (P * nmask)
        rem = s % (P * nmask)
        tmod = rem // nmask
        mask = rem % nmask
        if ci == goal_id:
            found = s
            break
        expanded += 1
        if budget and expanded > budget:
            return None, None, expanded, time.time() - st
        r, c = cells[ci]
        for a_i, a in enumerate(MOVES):
            dr, dc = DELTAS[a]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < Wd):
                continue
            sym = grid[nr][nc]
            if sym in (WALL, WATER):
                continue
            if a == W:
                cost = 1
            else:
                cost = m["swamp_cost"] if sym == SWAMP else 1
                climb = hs[nr][nc] - hs[r][c]
                if climb > 0:
                    cost += m["climb_cost"] * climb
            new_t = g + cost
            ntmod = new_t % P
            newmask = mask
            if sym == FORD:
                if ntmod >= K:
                    continue
            elif sym.isupper() and sym not in "SG":
                b = bidx[sym]
                if not (mask >> b) & 1:
                    continue
            if a != W and sym in winch_mask:
                newmask ^= winch_mask[sym]
            ns = st_of(cell_id[nr][nc], ntmod, newmask)
            ng = new_t
            if ng < dist[ns]:
                dist[ns] = ng
                parent[ns] = s
                pact[ns] = a_i
                push(heap, (ng + h[cell_id[nr][nc]], ng, ns))

    if found < 0:
        return None, expanded, len(dist), time.time() - st

    # reconstruct
    actions = []
    s = found
    while parent[s] >= 0:
        actions.append(MOVES[pact[s]])
        s = parent[s]
    actions.reverse()
    return "".join(actions), expanded, len(dist), time.time() - st


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "test-input.txt"
    out = sys.argv[2] if len(sys.argv) > 2 else "solution.txt"
    text = open(path, encoding="utf-8").read()
    route, expanded, total, elapsed = solve(text)
    if route is None:
        print("NO SOLUTION", expanded, elapsed)
        return 1
    open(out, "w").write(route)
    print("route len", len(route), "expanded", expanded, "states", total, "time %.2fs" % elapsed)
    # verify via the provided simulator if available
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)) or ".")
    try:
        import environment as E
        v = E.load_map(path)
        trace = E.simulate(v, route)
        print("valid run: %d minutes (%d actions)" % (trace[-1][0], len(trace) - 1))
    except Exception as e:
        print("verify skipped:", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())