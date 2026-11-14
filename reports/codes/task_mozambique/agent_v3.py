"""v3 agent: explicit boat roles (runner / supplier) + raw contracts."""
import json

INF = 10 ** 9


def prep(obs, memory):
    m = obs['map']
    n = len(m['roles'])
    routes = m['routes']
    dist = [[INF] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
        for j in range(n):
            if routes[i][j]:
                dist[i][j] = routes[i][j]
    for k in range(n):
        for i in range(n):
            dik = dist[i][k]
            if dik >= INF:
                continue
            for j in range(n):
                d = dik + dist[k][j]
                if d < dist[i][j]:
                    dist[i][j] = d
    nxt = [[-1] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            best = None
            for k in range(n):
                if k == i or not routes[i][k]:
                    continue
                cand = routes[i][k] + dist[k][j]
                if cand == dist[i][j] and (best is None or routes[i][k] < best[0]):
                    best = (routes[i][k], k)
            if best is not None:
                nxt[i][j] = best[1]
    memory.update(dist=dist, nxt=nxt, roles=m['roles'], home=m['homes'][obs['seat']],
                  opp_home=m['homes'][1 - obs['seat']], contracts=m['contracts'],
                  craft_idx=[i for i, c in enumerate(m['contracts']) if c['points'] > 5],
                  raw_idx=[i for i, c in enumerate(m['contracts']) if c['points'] <= 5],
                  timber=[i for i in range(n) if m['roles'][i] == 0],
                  fibre=[i for i in range(n) if m['roles'][i] == 1])
    home = memory['home']
    best = None
    for f in range(n):
        if m['roles'][f] != 2:
            continue
        nt = min(dist[f][i] for i in memory['timber'])
        nf = min(dist[f][i] for i in memory['fibre'])
        dests = sorted(dist[f][c['destination']] for c in m['contracts'] if c['points'] > 5)
        cost = dist[home][f] * 1.0 + 0.7 * (nt + nf) + 0.35 * sum(dests[:2]) + 0.15 * sum(dests)
        if best is None or cost < best[0]:
            best = (cost, f)
    memory['factory'] = best[1]
    memory['open_done'] = False
    memory['target'] = None


def nav(memory, pos, target):
    if pos == target:
        return target
    return memory['nxt'][pos][target]


def stock_of(obs, memory, site):
    st = obs['stocks'][site]
    if st is not None:
        return st
    return min(6, 4 + obs['round'] // 4)


def cnt(c):
    return c[0] + c[1] + c[2]


def crafted_target(obs, memory, site):
    """Earliest-deadline unclaimed crafted contract feasibly reachable."""
    dist = memory['dist']
    rnd = obs['round']
    best = None
    for i in memory['craft_idx']:
        if obs['claims'][i] != -1:
            continue
        c = memory['contracts'][i]
        d = dist[site][c['destination']]
        if rnd + d + 1 > c['deadline']:
            continue
        key = c['deadline'] * 10 + d
        if best is None or key < best[0]:
            best = (key, i, c['destination'], d)
    return best


def raw_target(obs, memory, site):
    dist = memory['dist']
    rnd = obs['round']
    role = memory['roles']
    best = None
    for i in memory['raw_idx']:
        if obs['claims'][i] != -1:
            continue
        c = memory['contracts'][i]
        g = c['good']
        q = c['quantity']
        # prefer our warehouse stock; else natural sites
        best_src = None
        for s in range(len(role)):
            if role[s] != g:
                continue
            have = obs['own']['warehouses'][s][g]
            st = stock_of(obs, memory, s)
            if have < q and st < q:
                continue
            d0 = 0 if s == site else dist[site][s]
            total = d0 + dist[s][c['destination']]
            if rnd + total + 2 > c['deadline']:
                continue
            if best_src is None or total < best_src[0]:
                best_src = (total, s, d0)
        if best_src is not None:
            key = best_src[0] + c['deadline'] * 0.02
            if best is None or key < best[0]:
                best = (key, i, best_src[1], c['destination'])
    return best


def source_for(obs, memory, F, good):
    role = memory['roles']
    dist = memory['dist']
    best = None
    for s in range(len(role)):
        if role[s] == good:
            st = stock_of(obs, memory, s)
            if st >= 1:
                key = (dist[F][s] if s != F else 0) - st * 0.2
                if best is None or key < best[0]:
                    best = (key, s)
    return best[1] if best else None


def runner_action(obs, memory, bi):
    own = obs['own']
    b = own['boats'][bi]
    if b['destination'] != -1 and b['eta'] > 0:
        return None
    pos, cargo = b['position'], b['cargo']
    F = memory['factory']
    home = memory['home']
    role = memory['roles']
    wh = own['warehouses']
    ws = own['workshops']
    jobs = own['jobs']
    rnd = obs['round']

    # crafted cargo -> deliver
    if cargo[2] > 0:
        tgt = crafted_target(obs, memory, pos)
        if tgt is not None:
            _, ci, dsite, d = tgt
            if pos == dsite:
                return (99, {'op': 'deliver', 'boat': bi, 'contract': ci})
            return (80, {'op': 'sail', 'boat': bi, 'destination': nav(memory, pos, dsite)})
        # nobody wants it: unload back at factory
        if pos == F:
            return (30, {'op': 'unload', 'boat': bi, 'good': 2, 'quantity': min(cargo[2], 3)})
        return (40, {'op': 'sail', 'boat': bi, 'destination': nav(memory, pos, F)})

    if not memory.get('open_done') and pos == home:
        if cargo[0] < 2 and wh[home][0] > 0 and cnt(cargo) < 3:
            q = min(2 - cargo[0], wh[home][0], 3 - cnt(cargo))
            if q > 0:
                return (86, {'op': 'load', 'boat': bi, 'good': 0, 'quantity': q})
        if cargo[1] < 1 and wh[home][1] > 0 and cnt(cargo) < 3:
            q = min(1 - cargo[1], wh[home][1], 3 - cnt(cargo))
            if q > 0:
                return (85, {'op': 'load', 'boat': bi, 'good': 1, 'quantity': q})
        if cargo[0] >= 2 and cargo[1] >= 1:
            return (83, {'op': 'sail', 'boat': bi, 'destination': nav(memory, pos, F)})
        return (12, {'op': 'sail', 'boat': bi, 'destination': nav(memory, pos, F)})

    if pos == F:
        if not ws[F] and wh[F][0] >= 2 and wh[F][1] >= 1:
            return (95, {'op': 'build_workshop', 'boat': bi})
        if ws[F] and wh[F][0] >= 2 and wh[F][1] >= 1 and crafted_target(obs, memory, F) is not None:
            if jobs[F] == 0:
                return (94, {'op': 'craft', 'boat': bi})
        if wh[F][2] > 0 and cnt(cargo) < 3 and crafted_target(obs, memory, F) is not None:
            return (93, {'op': 'load', 'boat': bi, 'good': 2, 'quantity': min(wh[F][2], 3 - cnt(cargo))})
        if cargo[0] or cargo[1]:
            if cargo[0] and wh[F][0] < 6:
                return (70, {'op': 'unload', 'boat': bi, 'good': 0, 'quantity': min(cargo[0], 6 - wh[F][0])})
            if cargo[1] and wh[F][1] < 6:
                return (69, {'op': 'unload', 'boat': bi, 'good': 1, 'quantity': min(cargo[1], 6 - wh[F][1])})
        # wait for an imminent output / craft, otherwise do raw contracts
        if crafted_target(obs, memory, F) is not None and (jobs[F] > 0 or wh[F][2] > 0):
            return (25, {'op': 'wait'})
        rt = raw_target(obs, memory, pos)
        if rt is not None:
            _, ci, src, dsite = rt
            c = memory['contracts'][ci]
            g = c['good']
            if pos == src:
                q = min(c['quantity'] - cargo[g], stock_of(obs, memory, pos), 3 - cnt(cargo))
                if q > 0:
                    return (60, {'op': 'gather', 'boat': bi, 'good': g, 'quantity': q})
            elif pos != src:
                return (50, {'op': 'sail', 'boat': bi, 'destination': nav(memory, pos, src)})
        return (20, {'op': 'wait'})

    # not at factory
    if pos == home and cnt(cargo) == 0 and memory.get('open_done'):
        # help with raw contracts from home stock
        rt = raw_target(obs, memory, pos)
        if rt is not None:
            _, ci, src, dsite = rt
            c = memory['contracts'][ci]
            g = c['good']
            if pos == src and wh[home][g] > 0:
                q = min(c['quantity'], wh[home][g], 3 - cnt(cargo))
                if q > 0:
                    return (60, {'op': 'load', 'boat': bi, 'good': g, 'quantity': q})
            return (50, {'op': 'sail', 'boat': bi, 'destination': nav(memory, pos, src)})
        return (15, {'op': 'sail', 'boat': bi, 'destination': nav(memory, pos, F)})

    # deliver raw contract if at destination and carrying
    for i in memory['raw_idx']:
        if obs['claims'][i] != -1:
            continue
        c = memory['contracts'][i]
        if c['good'] in (0, 1) and cargo[c['good']] >= c['quantity'] and pos == c['destination'] and rnd < c['deadline']:
            return (98, {'op': 'deliver', 'boat': bi, 'contract': i})
    g = role[pos]
    if g in (0, 1) and cnt(cargo) < 3 and (wh[F][g] + cargo[g] < (4 if g == 0 else 3)):
        st = stock_of(obs, memory, pos)
        q = min(3 - cnt(cargo), st)
        if q > 0:
            return (78, {'op': 'gather', 'boat': bi, 'good': g, 'quantity': q})
    if cnt(cargo):
        return (60, {'op': 'sail', 'boat': bi, 'destination': nav(memory, pos, F)})
    rt = raw_target(obs, memory, pos)
    if rt is not None:
        _, ci, src, dsite = rt
        if pos == src:
            c = memory['contracts'][ci]
            g2 = c['good']
            if g2 in (0, 1):
                q = min(c['quantity'] - cargo[g2], stock_of(obs, memory, pos), 3 - cnt(cargo))
                if q > 0:
                    return (70, {'op': 'gather', 'boat': bi, 'good': g2, 'quantity': q})
        else:
            return (45, {'op': 'sail', 'boat': bi, 'destination': nav(memory, pos, src)})
    return (25, {'op': 'sail', 'boat': bi, 'destination': nav(memory, pos, F)})


def supplier_action(obs, memory, bi):
    own = obs['own']
    b = own['boats'][bi]
    if b['destination'] != -1 and b['eta'] > 0:
        return None
    pos, cargo = b['position'], b['cargo']
    F = memory['factory']
    home = memory['home']
    role = memory['roles']
    wh = own['warehouses']
    ws = own['workshops']
    jobs = own['jobs']
    rnd = obs['round']
    C = cnt(cargo)

    # crafted cargo: deliver like runner
    if cargo[2] > 0:
        tgt = crafted_target(obs, memory, pos)
        if tgt is not None:
            _, ci, dsite, d = tgt
            if pos == dsite:
                return (99, {'op': 'deliver', 'boat': bi, 'contract': ci})
            return (80, {'op': 'sail', 'boat': bi, 'destination': nav(memory, pos, dsite)})
        if pos == F:
            return (30, {'op': 'unload', 'boat': bi, 'good': 2, 'quantity': min(cargo[2], 3)})
        return (40, {'op': 'sail', 'boat': bi, 'destination': nav(memory, pos, F)})

    # raw contract delivery when carrying matching goods
    for i in memory['raw_idx']:
        if obs['claims'][i] != -1:
            continue
        c = memory['contracts'][i]
        if c['good'] in (0, 1) and cargo[c['good']] >= c['quantity'] and pos == c['destination'] and rnd < c['deadline']:
            return (98, {'op': 'deliver', 'boat': bi, 'contract': i})

    if pos == F:
        # craft if possible (any boat can)
        if ws[F] and wh[F][0] >= 2 and wh[F][1] >= 1 and jobs[F] == 0 and crafted_target(obs, memory, F) is not None:
            return (94, {'op': 'craft', 'boat': bi})
        if not ws[F] and wh[F][0] >= 2 and wh[F][1] >= 1:
            return (95, {'op': 'build_workshop', 'boat': bi})
        if cargo[0] and wh[F][0] < 6:
            return (80, {'op': 'unload', 'boat': bi, 'good': 0, 'quantity': min(cargo[0], 6 - wh[F][0])})
        if cargo[1] and wh[F][1] < 6:
            return (79, {'op': 'unload', 'boat': bi, 'good': 1, 'quantity': min(cargo[1], 6 - wh[F][1])})
        # still need inputs?
        need_t = wh[F][0] < 2
        need_f = wh[F][1] < 1
        if need_t or need_f:
            g = 0 if need_t else 1
            src = source_for(obs, memory, F, g)
            if src is None:
                src = source_for(obs, memory, F, 1 - g)
            if src is not None and src != pos:
                return (55, {'op': 'sail', 'boat': bi, 'destination': nav(memory, pos, src)})
        # otherwise stockpile a bit / raw
        if wh[F][0] < 4:
            src = source_for(obs, memory, F, 0)
            if src is not None and src != pos:
                return (30, {'op': 'sail', 'boat': bi, 'destination': nav(memory, pos, src)})
        if wh[F][1] < 3:
            src = source_for(obs, memory, F, 1)
            if src is not None and src != pos:
                return (29, {'op': 'sail', 'boat': bi, 'destination': nav(memory, pos, src)})
        rt = raw_target(obs, memory, pos)
        if rt is not None:
            _, ci, src, dsite = rt
            if pos != src:
                return (26, {'op': 'sail', 'boat': bi, 'destination': nav(memory, pos, src)})
        return (20, {'op': 'wait'})

    if pos == home and not memory.get('open_done'):
        if cargo[0] < 2 and wh[home][0] > 0 and C < 3:
            q = min(2 - cargo[0], wh[home][0], 3 - C)
            if q > 0:
                return (85, {'op': 'load', 'boat': bi, 'good': 0, 'quantity': q})
        if cargo[1] < 1 and wh[home][1] > 0 and C < 3:
            q = min(1 - cargo[1], wh[home][1], 3 - C)
            if q > 0:
                return (84, {'op': 'load', 'boat': bi, 'good': 1, 'quantity': q})
        if cargo[0] >= 2 and cargo[1] >= 1:
            return (82, {'op': 'sail', 'boat': bi, 'destination': nav(memory, pos, F)})
        return (10, {'op': 'sail', 'boat': bi, 'destination': nav(memory, pos, F)})

    g = role[pos]
    if g in (0, 1) and C < 3:
        want = 4 if g == 0 else 3
        if wh[F][g] + cargo[g] < want:
            st = stock_of(obs, memory, pos)
            q = min(3 - C, st, want - cargo[g] - wh[F][g] + 1)
            if q > 0:
                return (72, {'op': 'gather', 'boat': bi, 'good': g, 'quantity': q})
    if C:
        return (60, {'op': 'sail', 'boat': bi, 'destination': nav(memory, pos, F)})
    rt = raw_target(obs, memory, pos)
    if rt is not None:
        _, ci, src, dsite = rt
        if pos == src:
            c = memory['contracts'][ci]
            g2 = c['good']
            if g2 in (0, 1):
                q = min(c['quantity'] - cargo[g2], stock_of(obs, memory, pos), 3 - C)
                if q > 0:
                    return (55, {'op': 'gather', 'boat': bi, 'good': g2, 'quantity': q})
        else:
            return (35, {'op': 'sail', 'boat': bi, 'destination': nav(memory, pos, src)})
    return (12, {'op': 'sail', 'boat': bi, 'destination': nav(memory, pos, F)})


def legal_ok(obs, act):
    for a in obs['legal_actions']:
        if a['op'] != act['op']:
            continue
        if a['op'] == 'wait':
            return a
        if all(a.get(k) == act.get(k) for k in ('boat', 'destination', 'good', 'quantity', 'contract')):
            return a
    return None


def agent(observation, memory):
    obs = observation
    if obs['round'] == 0 or 'dist' not in memory:
        prep(obs, memory)
    if not memory.get('open_done') and obs['round'] > 0:
        # opening is done once both boats have left home at least once
        if all(b['position'] != memory['home'] for b in obs['own']['boats']) or obs['round'] > 12:
            memory['open_done'] = True
    cands = []
    boats = obs['own']['boats']
    idle = memory.setdefault('idle', [0, 0, 0, 0, 0])
    for bi in range(len(boats)):
        if bi == 0:
            r = runner_action(obs, memory, bi)
        else:
            r = supplier_action(obs, memory, bi)
        if r is not None:
            cands.append((r[0] + 12 * idle[bi], bi, r[1]))
    cands.sort(key=lambda x: -x[0])
    chosen = None
    chosen_bi = None
    tried = set()
    for prio, bi, act in cands:
        tried.add(bi)
        a = legal_ok(obs, act)
        if a is not None:
            chosen = a
            chosen_bi = bi
            break
    for bi in range(len(boats)):
        if bi == chosen_bi:
            idle[bi] = 0
        elif bi in tried:
            idle[bi] = min(idle[bi] + 1, 8)
        else:
            idle[bi] = 0
    if chosen is None:
        a = legal_ok(obs, {'op': 'wait'})
        chosen = a if a is not None else obs['legal_actions'][0]
    return chosen
