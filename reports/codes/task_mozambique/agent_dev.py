"""Dev agent for 'The Professor of Maputo Bay' (revision 2).

Core plan: pick a manufacturing site (factory), ship 2T+1F from home with both
boats, build a workshop, craft, then keep a supply shuttle running between the
nearest timber/fibre sites and the factory and deliver crafted goods to crafted
contracts.  Raw contracts are picked up opportunistically.
"""
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
    memory['dist'] = dist
    memory['nxt'] = nxt
    memory['roles'] = m['roles']
    memory['home'] = m['homes'][obs['seat']]
    memory['opp_home'] = m['homes'][1 - obs['seat']]
    memory['contracts'] = m['contracts']
    memory['craft_idx'] = [i for i, c in enumerate(m['contracts']) if c['points'] > 5]
    memory['raw_idx'] = [i for i, c in enumerate(m['contracts']) if c['points'] <= 5]
    memory['timber_sites'] = [i for i in range(n) if m['roles'][i] == 0]
    memory['fibre_sites'] = [i for i in range(n) if m['roles'][i] == 1]
    home = memory['home']
    role = m['roles']
    best = None
    for f in range(n):
        if role[f] != 2:
            continue
        nt = min(dist[f][i] for i in memory['timber_sites'])
        nf = min(dist[f][i] for i in memory['fibre_sites'])
        dests = sorted(dist[f][c['destination']] for c in m['contracts'] if c['points'] > 5)
        cost = dist[home][f] * 1.0 + 0.7 * (nt + nf) + 0.35 * sum(dests[:2])
        if best is None or cost < best[0]:
            best = (cost, f)
    memory['factory'] = best[1]


def nav(memory, pos, target):
    if pos == target:
        return target
    return memory['nxt'][pos][target]


def stock_of(obs, memory, site):
    st = obs['stocks'][site]
    if st is not None:
        return st
    return min(6, 4 + obs['round'] // 4)


def cargo_count(c):
    return c[0] + c[1] + c[2]


def choose_crafted_dest(obs, memory, site):
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


def choose_raw_target(obs, memory, site):
    """Return (key, contract_idx, source_site, dest) if a raw contract is worth doing."""
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
        srcs = [s for s in range(len(role)) if role[s] == g]
        best_src = None
        for s in srcs:
            d0 = 0 if s == site else dist[site][s]
            st = stock_of(obs, memory, s)
            whs = obs['own']['warehouses'][s][g]
            if st + whs < q:
                continue
            total = d0 + dist[s][c['destination']]
            if rnd + total + 3 > c['deadline']:
                continue
            if best_src is None or total < best_src[0]:
                best_src = (total, s, d0)
        if best_src is not None:
            key = best_src[0] - c['deadline'] * 0.01
            if best is None or key < best[0]:
                best = (key, i, best_src[1], c['destination'])
    return best


def source_for(obs, memory, F, good):
    """Nearest site (natural or warehouse stock) that can supply `good`."""
    role = memory['roles']
    dist = memory['dist']
    best = None
    for s in range(len(role)):
        if role[s] == good:
            st = stock_of(obs, memory, s)
            if st >= 1:
                key = (dist[F][s] if s != F else 0) - st * 0.25
                if best is None or key < best[0]:
                    best = (key, s)
    return best[1] if best else None


def raw_mission(obs, memory, bi, pos, cargo, prio=58):
    dist = memory['dist']
    rnd = obs['round']
    role = memory['roles']
    assigns = memory.setdefault('raw_assign', [None] * 16)
    ai = assigns[bi]
    if ai is not None:
        c = memory['contracts'][ai]
        if obs['claims'][ai] != -1 or (rnd + dist[pos][c['destination']] + 1 > c['deadline'] and cargo[c['good']] < c['quantity']):
            assigns[bi] = None
        else:
            ai = assigns[bi]
            if ai is not None:
                c = memory['contracts'][ai]
                g = c['good']
                q = c['quantity']
                if pos == c['destination'] and cargo[g] >= q and rnd < c['deadline']:
                    return (prio + 8, {'op': 'deliver', 'boat': bi, 'contract': ai})
                if cargo[g] >= q:
                    return (prio, {'op': 'sail', 'boat': bi, 'destination': nav(memory, pos, c['destination'])})
                # find best source for g
                best = None
                for src in range(len(role)):
                    if role[src] != g:
                        continue
                    st = stock_of(obs, memory, src)
                    have = obs['own']['warehouses'][src][g]
                    if st + have < q - cargo[g]:
                        continue
                    key = (0 if src == pos else dist[pos][src]) + dist[src][c['destination']]
                    if best is None or key < best[0]:
                        best = (key, src)
                if best is None:
                    assigns[bi] = None
                else:
                    src = best[1]
                    if src == pos:
                        room = 3 - (cargo[0] + cargo[1] + cargo[2])
                        avail = stock_of(obs, memory, pos)
                        whs = obs['own']['warehouses'][pos][g]
                        qq = min(q - cargo[g], max(avail, whs), room)
                        if qq > 0:
                            if whs >= qq:
                                return (prio, {'op': 'load', 'boat': bi, 'good': g, 'quantity': qq})
                            return (prio, {'op': 'gather', 'boat': bi, 'good': g, 'quantity': qq})
                    return (prio - 5, {'op': 'sail', 'boat': bi, 'destination': nav(memory, pos, src)})
    # choose a new raw contract
    if cargo[0] or cargo[1] or cargo[2]:
        return None
    best = None
    for i in memory['raw_idx']:
        if obs['claims'][i] != -1:
            continue
        c = memory['contracts'][i]
        g = c['good']
        q = c['quantity']
        srcbest = None
        for src in range(len(role)):
            if role[src] != g:
                continue
            st = stock_of(obs, memory, src)
            if st + obs['own']['warehouses'][src][g] < q:
                continue
            total = (0 if src == pos else dist[pos][src]) + dist[src][c['destination']]
            if rnd + total + 2 > c['deadline']:
                continue
            if srcbest is None or total < srcbest[0]:
                srcbest = (total, src)
        if srcbest is not None:
            key = srcbest[0] + c['deadline'] * 0.05
            if best is None or key < best[0]:
                best = (key, i)
    if best is None:
        return None
    assigns[bi] = best[1]
    c = memory['contracts'][best[1]]
    g = c['good']
    srcbest = None
    for src in range(len(role)):
        if role[src] != g:
            continue
        st = stock_of(obs, memory, src)
        if st + obs['own']['warehouses'][src][g] < c['quantity']:
            continue
        total = (0 if src == pos else dist[pos][src]) + dist[src][c['destination']]
        if srcbest is None or total < srcbest[0]:
            srcbest = (total, src)
    src = srcbest[1]
    if src == pos:
        room = 3 - (cargo[0] + cargo[1] + cargo[2])
        qq = min(c['quantity'], max(stock_of(obs, memory, pos), obs['own']['warehouses'][pos][g]), room)
        if qq > 0:
            if obs['own']['warehouses'][pos][g] >= qq:
                return (prio, {'op': 'load', 'boat': bi, 'good': g, 'quantity': qq})
            return (prio, {'op': 'gather', 'boat': bi, 'good': g, 'quantity': qq})
    return (prio - 5, {'op': 'sail', 'boat': bi, 'destination': nav(memory, pos, src)})


def boat_action(obs, memory, bi):
    own = obs['own']
    boats = own['boats']
    if bi >= len(boats):
        return None
    b = boats[bi]
    if b['destination'] != -1 and b['eta'] > 0:
        return None
    pos = b['position']
    cargo = b['cargo']
    cnt = cargo_count(cargo)
    home = memory['home']
    F = memory['factory']
    role = memory['roles']
    dist = memory['dist']
    rnd = obs['round']
    wh = own['warehouses']
    ws = own['workshops']
    jobs = own['jobs']

    # ---------- crafted goods on board: deliver ----------
    if cargo[2] > 0:
        tgt = choose_crafted_dest(obs, memory, pos)
        if tgt is not None:
            _, ci, dsite, d = tgt
            if pos == dsite:
                return (95, {'op': 'deliver', 'boat': bi, 'contract': ci})
            return (60, {'op': 'sail', 'boat': bi, 'destination': nav(memory, pos, dsite)})
        if pos != F:
            return (40, {'op': 'sail', 'boat': bi, 'destination': nav(memory, pos, F)})
        return (1, {'op': 'wait'})

    # ---------- at home: fill 2 timber + 1 fibre ----------
    if pos == home and cargo[2] == 0:
        if cargo[0] < 2 and wh[home][0] > 0 and cnt < 3:
            q = min(2 - cargo[0], wh[home][0], 3 - cnt)
            if q > 0:
                return (66, {'op': 'load', 'boat': bi, 'good': 0, 'quantity': q})
        if cargo[1] < 1 and wh[home][1] > 0 and cnt < 3:
            q = min(1 - cargo[1], wh[home][1], 3 - cnt)
            if q > 0:
                return (65, {'op': 'load', 'boat': bi, 'good': 1, 'quantity': q})
        if cargo[0] >= 2 and cargo[1] >= 1:
            return (58, {'op': 'sail', 'boat': bi, 'destination': nav(memory, pos, F)})
        # nothing useful left at home: gather or raw
        rt = choose_raw_target(obs, memory, pos)
        if rt is not None:
            _, ci, src, dsite = rt
            if pos == src:
                c = memory['contracts'][ci]
                g = c['good']
                q = min(c['quantity'] - cargo[g], stock_of(obs, memory, src), 3 - cnt)
                if q > 0:
                    return (50, {'op': 'gather', 'boat': bi, 'good': g, 'quantity': q})
            return (20, {'op': 'sail', 'boat': bi, 'destination': nav(memory, pos, src)})
        return (10, {'op': 'sail', 'boat': bi, 'destination': nav(memory, pos, F)})

    # ---------- at factory ----------
    if pos == F:
        if not ws[F] and wh[F][0] >= 2 and wh[F][1] >= 1:
            return (92, {'op': 'build_workshop', 'boat': bi})
        if (ws[F] and wh[F][0] >= 2 and wh[F][1] >= 1 and jobs[F] == 0
                and choose_crafted_dest(obs, memory, F) is not None):
            return (90, {'op': 'craft', 'boat': bi})
        if wh[F][2] > 0 and cnt < 3 and choose_crafted_dest(obs, memory, F) is not None:
            q = min(wh[F][2], 3 - cnt)
            return (88, {'op': 'load', 'boat': bi, 'good': 2, 'quantity': q})
        # unload carried timber/fibre
        if cargo[0] > 0 and wh[F][0] < 6:
            q = min(cargo[0], 6 - wh[F][0])
            if q > 0:
                return (76, {'op': 'unload', 'boat': bi, 'good': 0, 'quantity': q})
        if cargo[1] > 0 and wh[F][1] < 6:
            q = min(cargo[1], 6 - wh[F][1])
            if q > 0:
                return (75, {'op': 'unload', 'boat': bi, 'good': 1, 'quantity': q})
        # fetch missing input
        if wh[F][0] < 2 or wh[F][1] < 1:
            target = source_for(obs, memory, F, 0) if wh[F][0] < 2 else source_for(obs, memory, F, 1)
            if target is None:
                target = source_for(obs, memory, F, 1) if wh[F][0] < 2 else source_for(obs, memory, F, 0)
            if target is not None and target != pos:
                return (62, {'op': 'sail', 'boat': bi, 'destination': nav(memory, pos, target)})
        # raw contract only once all crafted contracts are gone
        if choose_crafted_dest(obs, memory, F) is None:
            r = raw_mission(obs, memory, bi, pos, cargo, prio=58)
            if r is not None:
                return r
        return (1, {'op': 'wait'})

    # ---------- at a natural site: gather what the factory needs ----------
    g = role[pos]
    if g in (0, 1):
        want = 3 if g == 0 else 2
        need_factory = (wh[F][0] + cargo[0] < 4) if g == 0 else (wh[F][1] + cargo[1] < 3)
        if need_factory and cnt < 3:
            st = stock_of(obs, memory, pos)
            q = min(want - cargo[g], st, 3 - cnt)
            if q > 0:
                return (72, {'op': 'gather', 'boat': bi, 'good': g, 'quantity': q})
        # else maybe raw contract goods
        rt = choose_raw_target(obs, memory, pos)
        if rt is not None:
            _, ci, src, dsite = rt
            c = memory['contracts'][ci]
            gg = c['good']
            if pos == src and c['points'] <= 5:
                q = min(c['quantity'] - cargo[gg], stock_of(obs, memory, pos), 3 - cnt)
                if q > 0:
                    return (50, {'op': 'gather', 'boat': bi, 'good': gg, 'quantity': q})
        if cargo[0] >= 2 and cargo[1] >= 1:
            return (58, {'op': 'sail', 'boat': bi, 'destination': nav(memory, pos, F)})
        # go to the other supply site if we lack it, else factory
        if g == 0 and cargo[1] < 1 and cnt < 3:
            target = source_for(obs, memory, F, 1)
            if target is not None and target != pos:
                return (34, {'op': 'sail', 'boat': bi, 'destination': nav(memory, pos, target)})
        if g == 1 and cargo[0] < 2 and cnt < 3:
            target = source_for(obs, memory, F, 0)
            if target is not None and target != pos:
                return (34, {'op': 'sail', 'boat': bi, 'destination': nav(memory, pos, target)})
        return (45, {'op': 'sail', 'boat': bi, 'destination': nav(memory, pos, F)})

    # ---------- elsewhere with cargo ----------
    if cargo[0] or cargo[1]:
        # deliver raw contract if at destination with matching cargo
        for i in memory['raw_idx']:
            if obs['claims'][i] != -1:
                continue
            c = memory['contracts'][i]
            if c['good'] in (0, 1) and cargo[c['good']] >= c['quantity'] and pos == c['destination'] and rnd < c['deadline']:
                return (85, {'op': 'deliver', 'boat': bi, 'contract': i})
        return (45, {'op': 'sail', 'boat': bi, 'destination': nav(memory, pos, F)})

    # ---------- empty elsewhere ----------
    r = raw_mission(obs, memory, bi, pos, cargo, prio=58)
    if r is not None:
        return r
    return (10, {'op': 'sail', 'boat': bi, 'destination': nav(memory, pos, F)})


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
    cands = []
    for bi in range(len(obs['own']['boats'])):
        r = boat_action(obs, memory, bi)
        if r is not None:
            cands.append((r[0], bi, r[1]))
    cands.sort(key=lambda x: -x[0])
    chosen = None
    for prio, bi, act in cands:
        a = legal_ok(obs, act)
        if a is not None:
            chosen = a
            break
    if chosen is None:
        a = legal_ok(obs, {'op': 'wait'})
        chosen = a if a is not None else obs['legal_actions'][0]
    return chosen
