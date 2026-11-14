"""The Professor of Maputo Bay — solver.
Self-contained (stdlib only). Greedy logistics scheduler + reactive execution.
"""
import json, sys
from heapq import heappush, heappop

TIMBER, FIBRE, MANUF = 0, 1, 2

def dijkstra(routes, src):
    N = len(routes)
    dist = [10**9]*N
    prev = [-1]*N
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heappop(pq)
        if d != dist[u]:
            continue
        for v, w in enumerate(routes[u]):
            if w > 0 and d+w < dist[v]:
                dist[v] = d+w; prev[v] = u
                heappush(pq, (d+w, v))
    return dist, prev


class Sim:
    __slots__ = ('routes','roles','contracts','home','seat','boats','wh','ws','jobs',
                 'stocks','score','round','claims','claimed','crafted','dist','delivered','nh')
    def __init__(self, mapinfo, home, seat):
        self.routes = mapinfo['routes']; self.roles = mapinfo['roles']
        self.contracts = mapinfo['contracts']; self.home = home; self.seat = seat
        self.boats = [{'cargo':[0,0,0],'position':home,'destination':-1,'eta':0,'m':None} for _ in range(2)]
        self.wh = [[0,0,0] for _ in range(10)]; self.wh[home] = [5,3,0]
        self.ws = [False]*10; self.jobs = [0]*10
        self.stocks = [ (4 if r in (TIMBER,FIBRE) else 0) for r in self.roles ]
        self.score = 0; self.round = 0; self.claims = [-1]*len(self.contracts)
        self.claimed = [False]*len(self.contracts); self.crafted = 0
        self.dist = None; self.delivered = {}; self.nh = None
    def precompute(self):
        N = 10
        self.dist = [[0]*N for _ in range(N)]
        self.nh = [[-1]*N for _ in range(N)]
        for s in range(N):
            d, prev = dijkstra(self.routes, s)
            self.dist[s] = d
            for t in range(N):
                if t == s: continue
                if prev[t] == -1: continue
                cur = t
                while prev[cur] != s and prev[cur] != -1:
                    cur = prev[cur]
                self.nh[s][t] = cur
    def clone(self):
        s = Sim.__new__(Sim)
        s.routes=self.routes; s.roles=self.roles; s.contracts=self.contracts
        s.home=self.home; s.seat=self.seat
        s.boats=[dict(b) for b in self.boats]
        s.wh=[list(w) for w in self.wh]; s.ws=list(self.ws); s.jobs=list(self.jobs)
        s.stocks=list(self.stocks); s.score=self.score; s.round=self.round
        s.claims=list(self.claims); s.claimed=list(self.claimed); s.crafted=self.crafted
        s.dist=self.dist; s.nh=self.nh; s.delivered=dict(self.delivered)
        return s
    def advance(self):
        for b in self.boats:
            if b['eta'] > 0:
                b['eta'] -= 1
                if b['eta'] == 0:
                    b['position'] = b['destination']; b['destination'] = -1
        for site in range(10):
            if self.jobs[site] > 0:
                self.jobs[site] -= 1
                if self.jobs[site] == 0:
                    self.wh[site][MANUF] += 1
        self.round += 1
        if self.round % 4 == 0:
            for s in range(10):
                if self.roles[s] in (TIMBER,FIBRE) and self.stocks[s] < 6:
                    self.stocks[s] += 1
    def legal_actions(self):
        acts = [('wait',)]
        for bi, b in enumerate(self.boats):
            if b['eta'] > 0:
                continue
            pos = b['position']
            for j, w in enumerate(self.routes[pos]):
                if w > 0:
                    acts.append(('sail', bi, j))
            free = 3 - sum(b['cargo'])
            for g in range(3):
                av = self.wh[pos][g]
                for q in range(1, min(av, free)+1):
                    acts.append(('load', bi, g, q))
            role = self.roles[pos]
            if role in (TIMBER, FIBRE) and self.stocks[pos] > 0 and free > 0:
                for q in range(1, min(self.stocks[pos], free)+1):
                    acts.append(('gather', bi, role, q))
            for g in range(3):
                for q in range(1, b['cargo'][g]+1):
                    acts.append(('unload', bi, g, q))
            for ci, c in enumerate(self.contracts):
                if not self.claimed[ci] and c['destination'] == pos and b['cargo'][c['good']] >= c['quantity']:
                    acts.append(('deliver', bi, ci))
        boats_present = set(b['position'] for b in self.boats if b['eta'] == 0)
        for site in range(10):
            if self.roles[site] != MANUF or site not in boats_present:
                continue
            if not self.ws[site] and self.wh[site][0] >= 2 and self.wh[site][1] >= 1:
                acts.append(('build_workshop', site))
            elif self.ws[site] and self.wh[site][0] >= 2 and self.wh[site][1] >= 1:
                acts.append(('craft', site))
        if self.wh[self.home][0] >= 4 and self.wh[self.home][1] >= 2 and len(self.boats) < 3:
            acts.append(('build_boat',))
        return acts

    def apply(self, a):
        op = a[0]
        if op == 'wait': pass
        elif op == 'sail':
            _,bi,d = a; b=self.boats[bi]; b['destination']=d; b['eta']=self.routes[b['position']][d]
        elif op == 'load':
            _,bi,g,q = a; b=self.boats[bi]; self.wh[b['position']][g]-=q; b['cargo'][g]+=q
        elif op == 'gather':
            _,bi,g,q = a; b=self.boats[bi]; self.stocks[b['position']]-=q; b['cargo'][g]+=q
        elif op == 'unload':
            _,bi,g,q = a; b=self.boats[bi]; b['cargo'][g]-=q; self.wh[b['position']][g]+=q
        elif op == 'deliver':
            _,bi,ci = a; b=self.boats[bi]; c=self.contracts[ci]
            b['cargo'][c['good']]-=c['quantity']; self.score+=c['points']
            self.claimed[ci]=True; self.claims[ci]=self.seat; self.delivered[ci]=self.round
            if c['good']==MANUF: self.crafted+=1
        elif op == 'build_workshop':
            _,site = a; self.wh[site][0]-=2; self.wh[site][1]-=1; self.ws[site]=True
        elif op == 'craft':
            _,site = a; self.wh[site][0]-=2; self.wh[site][1]-=1; self.jobs[site]=2
        elif op == 'build_boat':
            self.wh[self.home][0]-=4; self.wh[self.home][1]-=2
            self.boats.append({'cargo':[0,0,0],'position':self.home,'destination':-1,'eta':0,'m':None})
        self.advance()
        return self


# ------------------------- greedy scheduler -------------------------
def manuf_sites(sim): return [s for s,r in enumerate(sim.roles) if r==MANUF]
def timber_sites(sim): return [s for s,r in enumerate(sim.roles) if r==TIMBER]
def fibre_sites(sim): return [s for s,r in enumerate(sim.roles) if r==FIBRE]

def unclaimed_range(sim, DL):
    return [ci for ci in range(len(sim.contracts)) if not sim.claimed[ci] and DL[ci] > sim.round]

def choose_ws(sim):
    ms = manuf_sites(sim)
    if not ms: return None
    ts = timber_sites(sim); fs = fibre_sites(sim)
    def cost(s):
        nt = min((sim.dist[s][t] for t in ts), default=0)
        nf = min((sim.dist[s][f] for f in fs), default=0)
        return sim.dist[sim.home][s] + 3*nt + 2*nf
    return min(ms, key=cost)


def sail_towards(sim, bi, dst):
    b = sim.boats[bi]
    return ('sail', bi, sim.nh[b['position']][dst])


def nearest_site(sim, pos, sites):
    # nearest site in `sites` to `pos`
    best = None; bd = 10**9
    for s in sites:
        if sim.dist[pos][s] < bd:
            bd = sim.dist[pos][s]; best = s
    return best, bd


def policy_action(sim, DL, ws):
    """Mission-committed deterministic scheduler (one action)."""
    R = sim.round
    C = sim.contracts
    n = len(C)
    ur = [ci for ci in range(n) if not sim.claimed[ci] and DL[ci] > R]
    ur_raw = [ci for ci in ur if C[ci]['good'] in (TIMBER, FIBRE)]
    ur_raw.sort(key=lambda ci: DL[ci])
    ur_manuf = [ci for ci in ur if C[ci]['good'] == MANUF]

    # global production actions (cheap, high value)
    for site in manuf_sites(sim):
        boats_at = any(b['eta']==0 and b['position']==site for b in sim.boats)
        if not boats_at: continue
        if not sim.ws[site] and sim.wh[site][0]>=2 and sim.wh[site][1]>=1 and ur_manuf:
            return ('build_workshop', site)
    for site in manuf_sites(sim):
        boats_at = any(b['eta']==0 and b['position']==site for b in sim.boats)
        if boats_at and sim.ws[site] and sim.wh[site][0]>=2 and sim.wh[site][1]>=1 and ur_manuf:
            return ('craft', site)

    # choose best per-boat action
    best = None
    for bi, b in enumerate(sim.boats):
        if b['eta'] > 0: continue
        a, prio, val = step_boat(sim, bi, DL, ur_raw, ur_manuf, ws)
        if a is None: continue
        key = (prio, val)
        if best is None or key > best[0]:
            best = (key, a)
    if best is not None:
        return best[1]
    return ('wait',)


def step_boat(sim, bi, DL, ur_raw, ur_manuf, ws):
    b = sim.boats[bi]
    pos = b['position']
    C = sim.contracts
    home = sim.home
    m = b.get('m')

    # ---- execute existing mission ----
    if m is not None:
        kind = m[0]
        if kind == 'deliver':
            ci = m[1]
            c = C[ci]
            if sim.claimed[ci] or DL[ci] <= sim.round:
                b['m'] = None
            elif pos == c['destination'] and b['cargo'][c['good']] >= c['quantity']:
                b['m'] = None
                return ('deliver', bi, ci), 1000, 0
            elif b['cargo'][c['good']] >= c['quantity']:
                return sail_towards(sim, bi, c['destination']), 200, -sim.dist[pos][c['destination']]
            else:
                # need to load/gather at a source
                return fetch(sim, bi, c['good'], c['quantity'], c['destination'], 200, DL[ci], prefer_home=True)
        elif kind == 'ferry':
            g, qty = m[1], m[2]
            if b['cargo'][g] >= qty:
                if pos == ws:
                    b['m'] = None
                    return ('unload', bi, g, qty), 100, 0
                return sail_towards(sim, bi, ws), 100, -sim.dist[pos][ws]
            else:
                return fetch(sim, bi, g, qty, ws, 100, 0, prefer_home=True)
        else:
            b['m'] = None

    # ---- assign new mission for empty docked boat ----
    # 1) ship produced manufactured goods
    if ur_manuf and sim.wh[pos][MANUF] > 0:
        ci = min(ur_manuf, key=lambda c: (DL[c], sim.dist[pos][C[c]['destination']]))
        b['m'] = ('deliver', ci)
        return step_boat(sim, bi, DL, ur_raw, ur_manuf, ws)
    # 2) raw contract delivery
    for ci in ur_raw:
        c = C[ci]; g = c['good']; q = c['quantity']
        if home_can(sim, g, q):
            t = sim.dist[pos][home] + 1 + sim.dist[home][c['destination']] + 1
            if sim.round + t <= DL[ci] - 1:
                b['m'] = ('deliver', ci)
                return step_boat(sim, bi, DL, ur_raw, ur_manuf, ws)
    for ci in ur_raw:
        c = C[ci]; g = c['good']; q = c['quantity']
        srcs = timber_sites(sim) if g==TIMBER else fibre_sites(sim)
        avail = [s for s in srcs if sim.stocks[s] >= q - sim.wh[home][g]]
        if not avail:
            avail = [s for s in srcs if sim.stocks[s] > 0]
        if not avail: continue
        ns = min(avail, key=lambda s: sim.dist[pos][s])
        t = sim.dist[pos][ns] + 1 + sim.dist[ns][c['destination']] + 1
        if sim.round + t <= DL[ci] - 1:
            b['m'] = ('deliver', ci)
            return step_boat(sim, bi, DL, ur_raw, ur_manuf, ws)
    # 3) manufacturing support (ferry raw materials to ws)
    if ws is not None and ur_manuf:
        return plan_ferry(sim, bi, ws, ur_raw)
    return None, 0, 0


def home_can(sim, g, q):
    return sim.wh[sim.home][g] >= q


def plan_ferry(sim, bi, ws, ur_raw):
    """Return (action, prio, val) to support manufacturing."""
    b = sim.boats[bi]; pos = b['position']; home = sim.home
    target_t = (2 if not sim.ws[ws] else 0) + 2
    target_f = (1 if not sim.ws[ws] else 0) + 1
    in_t = in_f = 0
    for bb in sim.boats:
        if bb.get('m') and bb['m'][0]=='ferry':
            g,q = bb['m'][1], bb['m'][2]
            if g==TIMBER: in_t += q
            else: in_f += q
    rem_t = max(0, target_t - sim.wh[ws][0] - in_t)
    rem_f = max(0, target_f - sim.wh[ws][1] - in_f)
    # choose good
    cands = []
    if rem_t > 0: cands.append((TIMBER, rem_t))
    if rem_f > 0: cands.append((FIBRE, rem_f))
    if not cands:
        return None, 0, 0
    cands.sort(key=lambda x: -(x[1]*(3 if x[0]==TIMBER else 2)))
    g, rem = cands[0]
    srcs = timber_sites(sim) if g==TIMBER else fibre_sites(sim)
    # source choice: home if it has surplus, else nearest natural site with stock
    if sim.wh[home][g] > 0:
        src = home; kind = 'load'
    else:
        avail = [s for s in srcs if sim.stocks[s] > 0]
        if not avail: return None, 0, 0
        src = min(avail, key=lambda s: sim.dist[pos][s]); kind = 'gather'
    q = min(3, rem)
    if kind=='load': q = min(q, sim.wh[home][g])
    else: q = min(q, sim.stocks[src])
    if q <= 0: return None, 0, 0
    b['m'] = ('ferry', g, q)
    return fetch(sim, bi, g, q, ws, 90, 0, prefer_home=False)


def fetch(sim, bi, g, q, dest, prio, val, prefer_home):
    """Move boat bi to a source of good g, gather/load q on arrival."""
    b = sim.boats[bi]; pos = b['position']; home = sim.home
    # decide source
    if prefer_home and sim.wh[home][g] >= q:
        src = home; via_warehouse = True
    elif sim.wh[home][g] > 0 and prefer_home and g == 0:
        src = home; via_warehouse = True
    else:
        srcs = timber_sites(sim) if g==TIMBER else fibre_sites(sim)
        avail = [s for s in srcs if sim.stocks[s] > 0]
        if not avail: return None, 0, 0
        src = min(avail, key=lambda s: sim.dist[pos][s]); via_warehouse = False
    if pos == src:
        q = min(q, sim.wh[src][g] if via_warehouse else sim.stocks[src])
        if q <= 0: return None, 0, 0
        op = 'load' if via_warehouse else 'gather'
        return (op, bi, g, q), prio, val
    return sail_towards(sim, bi, src), prio-20, val


def _carries_for_raw(sim, bb, ur_raw, g):
    for ci in ur_raw:
        if sim.contracts[ci]['good']==g and bb['cargo'][g] >= sim.contracts[ci]['quantity']:
            return True
    return False

def effective_score(sim, DL):
    sc = 0
    for ci, r in sim.delivered.items():
        if r < DL[ci]:
            sc += sim.contracts[ci]['points']
    return sc


def greedy_complete(sim0, DL, ws=None):
    sim = sim0.clone()
    if ws is None: ws = choose_ws(sim)
    for _ in range(MAXR):
        if sim.round >= MAXR: break
        a = policy_action(sim, DL, ws)
        sim.apply(a)
    return sim


MAXR = 80


def rollout_action(sim, DL, ws, lookahead=1):
    """1-ply rollout: try each legal action, greedy-complete, pick best."""
    acts = sim.legal_actions()
    if len(acts) <= 1:
        return acts[0] if acts else ('wait',)
    best_a = None
    best_v = -1
    for a in acts:
        ns = sim.clone(); ns.apply(a)
        res = greedy_complete(ns, DL, ws)
        v = res.score
        # tiebreak: prefer delivering / more deliveries
        if v > best_v or (v == best_v and best_a is None):
            best_v = v; best_a = a
    return best_a if best_a else ('wait',)


def prof_times(mapinfo, prof_home, seat):
    sim = Sim(mapinfo, prof_home, 1-seat); sim.precompute()
    DL = [c['deadline'] for c in mapinfo['contracts']]
    res = greedy_complete(sim, DL)
    return dict(res.delivered)


# ------------------------- agent glue -------------------------
def tuple_to_dict(action):
    op = action[0]
    if op == 'sail': _,b,d=action; return {'op':'sail','boat':b,'destination':d}
    if op == 'load': _,b,g,q=action; return {'op':'load','boat':b,'good':g,'quantity':q}
    if op == 'gather': _,b,g,q=action; return {'op':'gather','boat':b,'good':g,'quantity':q}
    if op == 'unload': _,b,g,q=action; return {'op':'unload','boat':b,'good':g,'quantity':q}
    if op == 'deliver': _,b,c=action; return {'op':'deliver','boat':b,'contract':c}
    if op in ('build_workshop','craft','build_boat','wait'): return {'op':op}
    return {'op':op}

def match_in_legal(ad, legal):
    for a in legal:
        if a.get('op') != ad['op']: continue
        if all(a.get(k)==v for k,v in ad.items() if k!='op'):
            return a
    return None

def map_sig(obs):
    m = obs['map']
    return (tuple(m['roles']), tuple(tuple(r) for r in m['routes']), tuple(m['homes']))

def sim_from_obs(obs):
    m = obs['map']; seat=obs['seat']; home=obs['own']['home']
    sim = Sim(m, home, seat); sim.precompute()
    o = obs['own']
    sim.boats=[dict(b) for b in o['boats']]
    sim.wh=[list(w) for w in o['warehouses']]
    sim.ws=list(o['workshops']); sim.jobs=list(o['jobs']); sim.score=o['score']
    sim.round=obs['round']; sim.crafted=o.get('crafted_deliveries',0)
    for s in range(10):
        v=obs['stocks'][s]
        if v is not None: sim.stocks[s]=v
        elif sim.roles[s] in (TIMBER,FIBRE): sim.stocks[s]=min(6,4+(sim.round//4))
        else: sim.stocks[s]=0
    for ci,cl in enumerate(obs['claims']):
        if cl!=-1:
            sim.claimed[ci]=True; sim.claims[ci]=cl
            if cl==seat: sim.delivered[ci]=0
    return sim


def sync_sim(sim, obs):
    """Update sim in place from a live observation, preserving boat missions."""
    o = obs['own']
    for i, bdict in enumerate(o['boats']):
        if i < len(sim.boats):
            b = sim.boats[i]
            b['cargo'] = list(bdict['cargo'])
            b['position'] = bdict['position']
            b['destination'] = bdict['destination']
            b['eta'] = bdict['eta']
        else:
            sim.boats.append({'cargo':list(bdict['cargo']),'position':bdict['position'],'destination':bdict['destination'],'eta':bdict['eta'],'m':None})
    # drop sailed-away extra boats beyond observation
    del sim.boats[len(o['boats']):]
    sim.wh = [list(w) for w in o['warehouses']]
    sim.ws = list(o['workshops'])
    sim.jobs = list(o['jobs'])
    sim.score = o['score']
    sim.round = obs['round']
    sim.crafted = o.get('crafted_deliveries', 0)
    for s in range(10):
        v = obs['stocks'][s]
        if v is not None: sim.stocks[s] = v
        elif sim.roles[s] in (TIMBER, FIBRE): sim.stocks[s] = min(6, 4 + (sim.round // 4))
        else: sim.stocks[s] = 0
    for ci, cl in enumerate(obs['claims']):
        if cl != -1:
            sim.claimed[ci] = True
            sim.claims[ci] = cl
            if cl == sim.seat and ci not in sim.delivered:
                sim.delivered[ci] = sim.round
    return sim


def agent(observation, memory):
    seat = observation['seat']
    rnd = observation['round']
    legal = observation['legal_actions']

    if rnd == 0 or memory.get('sig') != map_sig(observation):
        memory.clear()
        memory['sig'] = map_sig(observation)
        m = observation['map']; home = observation['own']['home']
        prof_home = m['homes'][1-seat]
        pt = prof_times(m, prof_home, seat)
        DL = []
        for ci,c in enumerate(m['contracts']):
            DL.append(min(c['deadline'], pt.get(ci, 10**9)))
        memory['DL'] = DL
        sim = Sim(m, home, seat); sim.precompute()
        memory['ws'] = choose_ws(sim)
        memory['sim'] = sim

    DL = memory['DL']
    ws = memory['ws']
    sim = sync_sim(memory['sim'], observation)
    a = policy_action(sim, DL, ws)
    d = tuple_to_dict(a)
    mm = match_in_legal(d, legal)
    if mm is not None:
        return mm
    # fallback
    for x in legal:
        if x.get('op')=='wait': return x
    return legal[0]


def serve():
    protocol = sys.stdout
    sys.stdout = sys.stderr
    memory = {}
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        observation = json.loads(line)['observation']
        protocol.write(json.dumps(agent(observation, memory)) + '\n')
        protocol.flush()

if __name__ == '__main__':
    serve()

# ------------------------- beam search planner -------------------------
def eval_state(sim, DL):
    v = float(sim.score)
    # manufactured goods ready to ship
    for s in range(10):
        if sim.roles[s] == MANUF:
            v += 12.0 * sim.wh[s][MANUF]
            if sim.ws[s]:
                craftable = min(sim.wh[s][0] // 2, sim.wh[s][1] // 1)
                v += 10.0 * craftable
                v += 1.5 * (sim.wh[s][0] + sim.wh[s][1])
            else:
                # materials staged to build workshop
                v += 1.0 * (sim.wh[s][0] + sim.wh[s][1])
            if sim.ws[s]:
                v += 6.0
    # goods on boats
    for b in sim.boats:
        c = b['cargo']
        v += 12.0 * c[MANUF]
        v += 1.2 * (c[TIMBER] + c[FIBRE])
        # small bonus for being near an unclaimed contract destination with matching goods
        if c[MANUF] > 0:
            for ci in range(len(sim.contracts)):
                if not sim.claimed[ci] and sim.contracts[ci]['good']==MANUF and DL[ci] > sim.round:
                    if b['eta']==0 and b['position']==sim.contracts[ci]['destination']:
                        v += 6.0
    # raw home stock reserve
    v += 0.5 * (sim.wh[sim.home][0] + sim.wh[sim.home][1])
    return v


def state_sig(sim):
    b = tuple((x['position'], x['eta'], x['destination'], tuple(x['cargo'])) for x in sim.boats)
    return (b, tuple(tuple(w) for w in sim.wh), tuple(sim.ws), tuple(sim.jobs),
            tuple(sim.stocks), sim.score, tuple(sim.claimed))


def beam_plan(mapinfo, home, seat, DL, width, time_budget):
    import time
    t0 = time.monotonic()
    sim = Sim(mapinfo, home, seat); sim.precompute()
    ws = choose_ws(sim)
    # nodes[i] = (sim, parent, action)
    nodes = [(sim, -1, None)]
    frontier = [0]
    for r in range(80):
        if time.monotonic() - t0 > time_budget:
            break
        cands = []
        for nid in frontier:
            st = nodes[nid][0]
            for act in st.legal_actions():
                ns = st.clone(); ns.apply(act)
                cands.append((ns, nid, act))
        if not cands:
            break
        best = {}
        for ns, nid, act in cands:
            sig = state_sig(ns)
            v = eval_state(ns, DL)
            if sig not in best or v > best[sig][0]:
                best[sig] = (v, nid, act, ns)
        ranked = sorted(best.values(), key=lambda t: -t[0])
        frontier = []
        for v, nid, act, ns in ranked[:width]:
            nodes.append((ns, nid, act))
            frontier.append(len(nodes)-1)
    best_nid = max(frontier, key=lambda i: eval_state(nodes[i][0], DL))
    seq = []
    cur = best_nid
    while nodes[cur][1] != -1:
        seq.append(nodes[cur][2])
        cur = nodes[cur][1]
    seq.reverse()
    # replay to get final sim
    res = Sim(mapinfo, home, seat); res.precompute()
    for a in seq:
        res.apply(a)
    return seq, res
