"""
Recover the Black Box — structure discovery solver (task_germany).

Hidden noise-free deterministic function family on [0,1]^5:

    f(x) = C + sum_i a_i*x_i
             + c*(x_p - 0.5)*(x_q - 0.5)      # exactly one centred-product interaction
             + J*[ g . x >= T ]               # exactly one linear step (jump J)

Approach (robust to the one-variable-at-a-time trap *and* to a step that
overlaps the smooth terms, because it never relies on a global smooth fit):

  Phase 1  finite-difference stencil (small h, several base points).
           Mixed second differences -> interaction pair + coefficient c;
           corrected forward differences -> linear coefficients a_i;
           both via the MEDIAN over base points, so a stencil that happens to
           straddle the step cannot corrupt the estimate.
  Phase 2  with the smooth part known, the residual r = f - smooth is exactly
           0 (off-step) or J (on-step): the step pops out cleanly. Corners are
           labelled jump/base, the gate SUPPORT is read by scanning cube edges.
  Phase 3  the switching hyperplane is reconstructed by bisection strictly
           inside the gate subspace (idle coords fixed to 0) and a PCA plane fit.
  Phase 4  the normal is oriented onto the jump side; the smooth coefficients
           are refit exactly on off-step points; predictions are produced.

Only numpy is used beyond the stdlib.
"""
import itertools
import numpy as np

VARS = ["x1", "x2", "x3", "x4", "x5"]
K = 5


def corners():
    return np.array(list(itertools.product([0.0, 1.0], repeat=K)))


def lhs(n, seed=20260827):
    rng = np.random.default_rng(seed)
    out = np.empty((n, K))
    for j in range(K):
        out[:, j] = (rng.permutation(n) + rng.random(n)) / n
    return out


def feats(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    C = X - 0.5
    cols = [np.ones(len(X))]
    for j in range(K):
        cols.append(X[:, j])
    for p in range(K):
        for q in range(p + 1, K):
            cols.append(C[:, p] * C[:, q])
    return np.column_stack(cols)


class _Query:
    def __init__(self, oracle):
        self.oracle = oracle

    def many(self, xs):
        return self.oracle.ask_many([list(map(float, x)) for x in xs])

    def one(self, x):
        return self.oracle.ask(list(map(float, x)))


class Solver:
    def __init__(self, oracle, points):
        self.q = _Query(oracle)
        self.oracle = oracle
        self.points = np.asarray(points, dtype=float)

    def _model(self, X, a, c0, pair, c, g, T, J):
        X = np.asarray(X, float)
        y = np.full(len(X), c0) + X @ a
        if pair is not None:
            p, q = int(pair[0]), int(pair[1])
            y = y + c * (X[:, p] - 0.5) * (X[:, q] - 0.5)
        y = y + J * (X @ g >= T).astype(float)
        return y

    # ------------------------------------------------------------------
    def run(self):
        h = 0.05
        m = 4
        # base points spread so the whole stencil lives in [0,1]
        B = lhs(m, seed=20260827) * (1.0 - 2 * h - 0.0) + h * 0.1
        B = np.clip(B, 0.02, 1.0 - 2 * h - 0.02)

        # stencil: base, base+h ei (5), base+h ei+h ej (10) per base point
        pts = []
        tags = []
        for k in range(m):
            x = B[k]
            pts.append(x); tags.append(("b", k))
            for i in range(K):
                e = np.zeros(K); e[i] = h
                pts.append(x + e); tags.append(("f", k, i))
            for i in range(K):
                for j in range(i + 1, K):
                    e = np.zeros(K); e[i] = h; e[j] = h
                    pts.append(x + e); tags.append(("ff", k, i, j))
        ys = np.array(self.q.many(pts))

        fb = np.zeros(m)
        ff = np.zeros((m, K))
        fff = np.zeros((m, K, K))
        for (t, y) in zip(tags, ys):
            if t[0] == "b":
                fb[t[1]] = y
            elif t[0] == "f":
                ff[t[1], t[2]] = y
            else:
                fff[t[1], t[2], t[3]] = y

        # ---- interaction pair + c (small-magnitude = safe stencil cluster) ----
        D = {}
        Dval = {}
        for i in range(K):
            for j in range(i + 1, K):
                d = fff[:, i, j] - ff[:, i] - ff[:, j] + fb
                safe = d[np.abs(d) < 0.5]
                D[(i, j)] = float(np.median(safe)) if len(safe) else float(np.median(d))
        pair = max(D, key=lambda k: abs(D[k]))
        c = D[pair] / (h * h)

        # ---- linear coefficients a_i (interaction-corrected, safe cluster) ----
        a = np.zeros(K)
        for i in range(K):
            vals = np.zeros(m)
            for k in range(m):
                v = (ff[k, i] - fb[k]) / h
                if i == pair[0]:
                    v -= c * (B[k, pair[1]] - 0.5)
                elif i == pair[1]:
                    v -= c * (B[k, pair[0]] - 0.5)
                vals[k] = v
            safe = vals[np.abs(vals) < 5.0]
            a[i] = float(np.median(safe)) if len(safe) else float(np.median(vals))

        # ---- intercept C (median; possibly offset by J, fixed after boundary) ----
        Cvals = np.zeros(m)
        for k in range(m):
            x = B[k]
            v = fb[k] - float(x @ a)
            if abs(c) > 1e-12:
                v -= c * (x[pair[0]] - 0.5) * (x[pair[1]] - 0.5)
            Cvals[k] = v
        c0 = float(np.median(Cvals))

        # ---- clean residual -> corner labels ----
        corners_all = corners()
        yc = np.array(self.q.many(corners_all))
        gr = lambda X: (np.full(len(X), c0) + X @ a
                        + (c * (X[:, pair[0]] - 0.5) * (X[:, pair[1]] - 0.5)
                           if abs(c) > 1e-12 else 0.0))
        rcorner = yc - gr(corners_all)
        lo, hi = np.median(rcorner), np.max(rcorner)
        # split residual into two clusters
        c_lo, c_hi = self._two_means(rcorner)
        c_lo, c_hi = (c_lo, c_hi) if c_lo < c_hi else (c_hi, c_lo)
        jump = c_hi - c_lo
        mid = 0.5 * (c_lo + c_hi)
        is_jump = rcorner > mid

        if jump < 1e-6 or (not is_jump.any()) or is_jump.all():
            return self._degrade(a, c0, pair, c)

        # ---- gate support via edge scan (corner level) ----
        support = []
        for j in range(K):
            bit = 1 << (4 - j)
            xor_idx = np.arange(len(corners_all)) ^ bit
            if bool((is_jump != is_jump[xor_idx]).any()):
                support.append(j)
        if not support:
            return self._degrade(a, c0, pair, c)

        # ---- expand support with a boundary-point probe (catches gate
        # variables whose corner-level effect is masked by other gates) ----
        support = self._probe_expand(corners_all, is_jump, support, gr, mid)

        g, T = self._reconstruct(corners_all, is_jump, support, c_lo, mid,
                                 gr, c, pair)

        # orient g onto the jump side
        ji = np.where(is_jump)[0][0]
        ni = np.where(~is_jump)[0][0]
        if float(g @ corners_all[ji] - T) < float(g @ corners_all[ni] - T):
            g = -g
            T = -T

        # ---- fix intercept & jump using off-step (base) points ----
        X1 = np.vstack([np.array(pts), corners_all])
        y1 = np.concatenate([ys, yc])
        sd = X1 @ g - T
        base = sd < -0.005
        if base.sum() < 2:
            base = sd < 0.0
        gb = lambda X: (np.full(len(X), 0.0) + X @ a
                        + (c * (X[:, pair[0]] - 0.5) * (X[:, pair[1]] - 0.5)
                           if abs(c) > 1e-12 else 0.0))
        bvals = y1[base] - gb(X1[base])
        c0 = float(np.median(bvals))
        # exact residual -> jump (sign-agnostic)
        rall = y1 - (c0 + gb(X1))
        c_lo, c_hi = self._two_means(rall)
        c_lo, c_hi = (c_lo, c_hi) if c_lo < c_hi else (c_hi, c_lo)
        jump = c_hi - c_lo

        # ---- assemble ----
        relv = set(int(j) for j in range(K) if abs(a[j]) > 0.05)
        if pair is not None:
            relv.add(int(pair[0])); relv.add(int(pair[1]))
        relv |= set(int(j) for j in support)
        relevant = [VARS[j] for j in sorted(relv)]
        irrelevant = [VARS[j] for j in range(K) if j not in relv]

        inter = []
        if pair is not None:
            inter = [{"variables": [VARS[int(pair[0])], VARS[int(pair[1])]],
                      "type": "product", "coefficient": float(c),
                      "equivalent_centered_form": (
                          f"{c:.6f}*({VARS[int(pair[0])]}-0.5)"
                          f"*({VARS[int(pair[1])]}-0.5)")}]

        coeff_map = {VARS[j]: float(g[j]) for j in range(K)}
        disc = [{"gate_variables": [VARS[j] for j in support],
                 "boundary": {"form": "linear", "coefficients": coeff_map,
                              "threshold": float(T),
                              "active_when": "greater_equal"},
                 "jump": abs(float(jump)),
                 "condition": self._cond(g, T)}]

        expr = self._expr(a, c0, pair, c, g, T, jump)
        preds = self._model(self.points, a, c0, pair, c, g, T, jump)

        return {
            "relevant_variables": relevant,
            "irrelevant_variables": irrelevant,
            "interactions": inter,
            "discontinuities": disc,
            "model_expression": expr,
            "predictions": [round(float(p), 6) for p in preds],
            "notes": ("queries={:d}; finite-difference stencil + corner labels; "
                      "gate support {:s}; linear={:s}; interaction={:s} "
                      "c={:.5f}; jump={:.5f}.".format(
                          self.oracle.spent, str([VARS[j] for j in support]),
                          ", ".join(f"{VARS[j]}={a[j]:.4f}" for j in range(K)),
                          (f"{VARS[int(pair[0])]}*{VARS[int(pair[1])]}"
                           if pair else "none"), c, jump)),
        }

    # ------------------------------------------------------------------
    def _two_means(self, r):
        r = np.asarray(r, float)
        lo, hi = float(np.min(r)), float(np.max(r))
        if hi - lo < 1e-12:
            return lo, hi
        for _ in range(400):
            grp = np.abs(r - lo) <= np.abs(r - hi)
            if grp.all() or (~grp).all():
                break
            nlo = float(r[grp].mean())
            nhi = float(r[~grp].mean())
            if abs(nlo - lo) + abs(nhi - hi) < 1e-15:
                break
            lo, hi = nlo, nhi
        return lo, hi

    def _pair_idx(self, k):
        cnt = 0
        for p in range(K):
            for q in range(p + 1, K):
                if cnt == k:
                    return (p, q)
                cnt += 1
        return None

    def _sid(self, x, gr, mid):
        return float(self.q.one(x) - gr(np.asarray([x], float))[0]) > mid

    def _probe_expand(self, corners_all, is_jump, support, gr, mid):
        # a central boundary point P0
        P0 = self._central_boundary(corners_all, is_jump, gr, mid)
        if P0 is None:
            return support
        sup = set(support)
        d = 0.1
        for j in range(K):
            e = np.zeros(K); e[j] = d
            pp = np.clip(P0 + e, 0.0, 1.0)
            pm = np.clip(P0 - e, 0.0, 1.0)
            lp = self._sid(pp, gr, mid)
            lm = self._sid(pm, gr, mid)
            if lp != lm:
                sup.add(j)
        return sorted(sup)

    def _central_boundary(self, corners_all, is_jump, gr, mid):
        act = np.where(is_jump)[0]
        ina = np.where(~is_jump)[0]
        pairs = []
        for bi in ina:
            for ai in act:
                pairs.append((bi, ai))
        pairs.sort(key=lambda ab: np.sum(corners_all[ab[0]] != corners_all[ab[1]]),
                   reverse=True)
        for bi, ai in pairs:
            if self.oracle.remaining < 12:
                break
            p = self._bisect(corners_all[bi], corners_all[ai],
                             lambda x: self._sid(x, gr, mid), 9)
            if p is not None:
                return p
        return None

    def _reconstruct(self, corners_all, is_jump, support, c_lo, mid, gr, c, pair):
        # support corners = corners with idle coords set to 0
        mask = np.ones(K, bool)
        mask[support] = False
        is_sc = np.all(corners_all[:, mask] == 0.0, axis=1)
        sc_idx = np.where(is_sc)[0]
        act = [i for i in sc_idx if is_jump[i]]
        ina = [i for i in sc_idx if not is_jump[i]]
        pairs = []
        for bi in ina:
            for ai in act:
                pairs.append((bi, ai))
        pairs.sort(key=lambda ab: np.sum(corners_all[ab[0]] != corners_all[ab[1]]),
                   reverse=True)

        bpts, seen = [], set()
        budget = self.oracle.remaining
        iters = 8
        k = len(support)
        want = k + 2
        for bi, ai in pairs:
            if len(bpts) >= want or (self.oracle.remaining - iters) < 2:
                break
            p = self._bisect(corners_all[bi], corners_all[ai],
                             lambda x: self._sid(x, gr, mid), iters)
            if p is not None:
                key = tuple(np.round(p, 4))
                if key not in seen:
                    seen.add(key)
                    bpts.append(p)

        if k == 1:
            if not bpts:
                return self._unit_e(support[0]), 0.0
            P = np.asarray(bpts[0], float)
            g = self._unit_e(support[0])
            return g, float(P[support[0]])

        if len(bpts) < 2:
            # degenerate: not enough boundary points in the support subspace;
            # fall back to the largest-principal-component direction
            if len(bpts) == 1:
                P = np.asarray(bpts, float)
                g = np.zeros(K)
                g[support[0]] = 1.0
                return g, float(P[0, support[0]])
            return self._unit_e(support[0]), 0.0

        P = np.asarray(bpts, float)
        Ps = P[:, support] - P[:, support].mean(axis=0)
        U, s, Vt = np.linalg.svd(Ps)
        gs = Vt[-1]
        g = np.zeros(K)
        g[support] = gs
        T = float(gs @ P[0, support])
        return g, T

    def _unit_e(self, j):
        e = np.zeros(K)
        e[j] = 1.0
        return e

    def _bisect(self, a, b, side_of, iters=8):
        a = np.asarray(a, float); b = np.asarray(b, float)
        sa = side_of(a); sb = side_of(b)
        if sa == sb:
            return None
        lo, hi = a, b
        for _ in range(iters):
            m = 0.5 * (lo + hi)
            if side_of(m) == sa:
                lo = m
            else:
                hi = m
        return 0.5 * (lo + hi)

    def _cond(self, g, T):
        terms = [f"{g[j]:.6f}*{VARS[j]}" for j in range(K) if abs(g[j]) > 1e-10]
        return " + ".join(terms) + f" >= {T:.6f}"

    def _expr(self, a, c0, pair, c, g, T, J):
        parts = [f"{c0:.6f}"]
        for j in range(K):
            if abs(a[j]) > 1e-12:
                parts.append(f"{a[j]:.6f}*{VARS[j]}")
        if pair is not None:
            p, q = int(pair[0]), int(pair[1])
            parts.append(f"{c:.6f}*({VARS[p]}-0.5)*({VARS[q]}-0.5)")
        parts.append(f"({J:.6f} if {self._cond(g, T)} else 0.0)")
        return " + ".join(parts)

    def _degrade(self, a, c0, pair, c):
        X = self.points
        y = np.full(len(X), c0) + X @ a
        if pair is not None:
            p, q = int(pair[0]), int(pair[1])
            y = y + c * (X[:, p] - 0.5) * (X[:, q] - 0.5)
        rel = [VARS[j] for j in range(K) if abs(a[j]) > 0.05]
        irr = [VARS[j] for j in range(K) if abs(a[j]) <= 0.05]
        return {"relevant_variables": rel, "irrelevant_variables": irr,
                "interactions": [], "discontinuities": [],
                "model_expression": "",
                "predictions": [round(float(p), 6) for p in y],
                "notes": "degraded: smooth model only"}


def solve(oracle, points):
    return Solver(oracle, points).run()