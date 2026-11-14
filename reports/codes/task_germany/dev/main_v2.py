"""Black Box -- recover the wiring diagram of a hidden 5-dial device.

Family (the public practice box and the graded box are stated to be from the
same family):

    f(x) = b0 + sum_i b_i x_i + c*(x_a-0.5)*(x_b-0.5) + J*1[w . x >= T]
         = smooth quadratic (one bilinear term)  +  one hard step across a
           linear switching surface.

Strategy (<= 150 readings):

  1. All 32 corners of [0,1]^5.  Any half-space that intersects the cube
     contains a corner on each side (the max/min of w.x over the cube are
     corners), so this *guarantees* readings from both sides even when the
     switched cap is tiny -- the case that defeats one-dial-at-a-time probing.
  2. Six low-discrepancy base points with five one-sided finite differences
     each.  The gradient of the smooth part is affine, so ~30 gradient
     equations determine the whole quadratic (main effects + every bilinear
     term).  A step only spoils the rare probe that straddles it (the error is
     J/h, hundreds of times larger than the signal); deterministic RANSAC over
     4-base subsets throws those away.
  3. With the smooth part known, u = y - smooth(x) is two-valued: every
     reading is labelled inside/outside the switched region for free.
  4. Boundary points are found by bisecting segments between readings on
     opposite sides.  Six well-spread crossings determine the switching
     hyper-plane (total least squares); the dials with non-zero normal are the
     gate variables, the rest are dropped for sparsity.
  5. One exact least squares on (1, x_i, x_i x_j, 1[gate]) over every reading
     re-estimates all coefficients and the jump, both gate orientations are
     tried, and the exact expression + 300 predictions are emitted.
"""
import os
import sys

import numpy as np

VARS = ("x1", "x2", "x3", "x4", "x5")
PRIMES = (2, 3, 5, 7, 11)
H = 0.004                      # finite-difference step
PAIR_LIST = [(i, j) for i in range(5) for j in range(i + 1, 5)]
BKEYS = [(i, j) for i in range(5) for j in range(i, 5)]
BCOL = {k: 5 + n for n, k in enumerate(BKEYS)}


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _vdc(i, b):
    """i-th van der Corput coordinate in base b (i >= 1)."""
    f, r = 1.0 / b, 0.0
    while i > 0:
        r += f * (i % b)
        i //= b
        f /= b
    return r


def _halton(n, skip=0, lo=0.0, hi=1.0):
    P = np.empty((n, 5))
    for k in range(n):
        for j in range(5):
            P[k, j] = _vdc(skip + k + 1, PRIMES[j])
    return lo + (hi - lo) * P


def _grad_system(rows):
    """Design matrix / rhs for the 20-parameter affine gradient field.

    theta = [a_0..a_4, B_00, B_01, ..., B_44]  (B symmetric, 15 unique)
    A row for direction i at base x is
        a_i + sum_j B_ij x_j + (s h / 2) B_ii
    (the last term is the one-sided finite-difference bias).
    """
    n = len(rows)
    A = np.zeros((n, 20))
    g = np.zeros(n)
    for r, (x, i, s, gi) in enumerate(rows):
        A[r, i] = 1.0
        for j in range(5):
            key = (i, j) if i <= j else (j, i)
            A[r, BCOL[key]] += x[j]
        A[r, BCOL[(i, i)]] += 0.5 * s * H
        g[r] = gi
    return A, g


def _robust_grad_fit(rows, nbase):
    """RANSAC over 4-base subsets, then iterated reweighting on inliers."""
    A, g = _grad_system(rows)
    n = len(rows)
    tol = 0.01
    best = None
    if nbase >= 4 and n >= 20:
        from itertools import combinations
        for comb in combinations(range(nbase), 4):
            idx = [b * 5 + k for b in comb for k in range(5)]
            if max(idx) >= n:
                continue
            try:
                th, *_ = np.linalg.lstsq(A[idx], g[idx], rcond=None)
            except np.linalg.LinAlgError:
                continue
            inl = int((np.abs(A @ th - g) < tol).sum())
            if best is None or inl > best[0]:
                best = (inl, th)
    if best is None:
        theta, *_ = np.linalg.lstsq(A, g, rcond=None)
        return theta, np.ones(n, bool), 0
    theta = best[1]
    for _ in range(8):
        keep = np.abs(A @ theta - g) < tol
        if keep.sum() < 20 or keep.all():
            break
        new, *_ = np.linalg.lstsq(A[keep], g[keep], rcond=None)
        if np.array_equal(new, theta):
            theta = new
            break
        theta = new
    keep = np.abs(A @ theta - g) < tol
    return theta, keep, int(keep.sum())


def _split_two(u, min_size=3):
    """Find the best two-level split of u.  Returns (lo, hi, labels) or None."""
    n = len(u)
    if n < 2 * min_size:
        return None
    order = np.argsort(u)
    us = u[order]
    gaps = np.diff(us)
    for kg in np.argsort(gaps)[::-1][:8]:
        kg = int(kg)
        if kg + 1 < min_size or n - kg - 1 < min_size:
            continue
        gap = float(gaps[kg])
        if gap < 0.02:
            continue
        lo, hi = us[:kg + 1], us[kg + 1:]
        lo_v = float(np.median(lo))
        hi_v = float(np.median(hi))
        spread = max(lo[-1] - lo[0], hi[-1] - hi[0])
        if spread > max(0.02, 0.10 * gap):
            continue
        lab = u > 0.5 * (lo_v + hi_v)
        if lab.sum() < min_size or (~lab).sum() < min_size:
            continue
        return lo_v, hi_v, lab
    return None


def _fps(X, k, start=None):
    """farthest-point sample of k indices from X."""
    n = len(X)
    if n <= k:
        return list(range(n))
    if start is None:
        c = X.mean(axis=0)
        start = int(np.argmax(((X - c) ** 2).sum(axis=1)))
    sel = [start]
    d = ((X - X[start]) ** 2).sum(axis=1)
    for _ in range(k - 1):
        i = int(np.argmax(d))
        sel.append(i)
        d = np.minimum(d, ((X - X[i]) ** 2).sum(axis=1))
    return sel


# ----------------------------------------------------------------------------
def solve(oracle, points):
    Xq = np.asarray(points, dtype=np.float64).reshape(-1, 5)
    budget0 = int(getattr(oracle, "remaining", 150))
    data = {}
    log = []

    # --- query bookkeeping ---------------------------------------------------
    def key(x):
        return tuple(round(float(v), 9) for v in x)

    def ask_many(Xs, phase=None):
        Xs = np.clip(np.asarray(Xs, dtype=np.float64), 0.0, 1.0).reshape(-1, 5)
        out = [None] * len(Xs)
        todo, pos = [], []
        for k, x in enumerate(Xs):
            kk = key(x)
            if kk in data:
                out[k] = data[kk][1]
            else:
                todo.append(x)
                pos.append(k)
        room = max(0, int(oracle.remaining) - 1)
        if len(todo) > room:
            todo, pos = todo[:room], pos[:room]
        if todo:
            ys = oracle.ask_many([[float(v) for v in x] for x in todo], phase)
            for x, y in zip(todo, ys):
                data[key(x)] = (np.asarray(x, float), float(y))
            for k, y in zip(pos, ys):
                out[k] = float(y)
        return out

    def ask(x, phase=None):
        return ask_many([x], phase)[0]

    def snapshot():
        ks = list(data.keys())
        X = np.array([data[k][0] for k in ks], float)
        y = np.array([data[k][1] for k in ks], float)
        return X, y

    # ------------------------------------------------------------------
    # 1. all 32 corners: guarantee readings on both sides of any half-space
    # ------------------------------------------------------------------
    corners = np.array([[float(i), float(j), float(k), float(l), float(m)]
                        for i in (0.0, 1.0) for j in (0.0, 1.0)
                        for k in (0.0, 1.0) for l in (0.0, 1.0)
                        for m in (0.0, 1.0)])
    yc = ask_many(corners, "corners")
    log.append("corners:%d" % sum(1 for y in yc if y is not None))

    # ------------------------------------------------------------------
    # 2. finite-difference gradients at low-discrepancy bases
    # ------------------------------------------------------------------
    nbase = 6
    bases = _halton(nbase, skip=3, lo=0.06, hi=0.94)
    rows = []
    for b in bases:
        pts = [b]
        signs = []
        for i in range(5):
            s = 1.0 if b[i] + H <= 1.0 else -1.0
            p = b.copy()
            p[i] += s * H
            pts.append(p)
            signs.append(s)
        ys = ask_many(pts, "grad")
        if ys[0] is None:
            continue
        for i in range(5):
            if ys[i + 1] is None:
                continue
            rows.append((b.copy(), i, signs[i],
                         signs[i] * (ys[i + 1] - ys[0]) / H))

    # The family has ONE bilinear term, so the gradient field is sparse:
    #   g_i(x) = a_i + c*x_j  (i == one side of the pair), a_i otherwise.
    # Fitting that 7-parameter structure per candidate pair is far better
    # conditioned than a general 20-parameter quadratic.
    def sparse_grad_fit(rows):
        n = len(rows)
        if n < 8:
            return None
        Adir = np.zeros((n, 5))
        g = np.empty(n)
        for r, (x, i, s, gi) in enumerate(rows):
            Adir[r, i] = 1.0
            g[r] = gi
        rng = np.random.default_rng(20261114)
        best = None
        for (a_, b_) in PAIR_LIST:
            col = np.empty(n)
            for r, (x, i, s, gi) in enumerate(rows):
                if i == a_:
                    col[r] = x[b_]
                elif i == b_:
                    col[r] = x[a_]
                else:
                    col[r] = 0.0
            A = np.column_stack([Adir, col])
            cand = None
            for _ in range(300):
                idx = rng.choice(n, 8, replace=False)
                try:
                    th, *_ = np.linalg.lstsq(A[idx], g[idx], rcond=None)
                except np.linalg.LinAlgError:
                    continue
                inl = int((np.abs(A @ th - g) < 0.01).sum())
                if cand is None or inl > cand[0]:
                    cand = (inl, th)
            if cand is None:
                continue
            th = cand[1]
            for _ in range(6):
                keep = np.abs(A @ th - g) < 0.01
                if keep.sum() < 8:
                    break
                th2, *_ = np.linalg.lstsq(A[keep], g[keep], rcond=None)
                if np.array_equal(th2, th):
                    th = th2
                    break
                th = th2
            keep = np.abs(A @ th - g) < 0.01
            inl = int(keep.sum())
            score = (inl, abs(float(th[5])))
            if best is None or score > best[0]:
                best = (score, (a_, b_), th, keep)
        return best

    fitg = sparse_grad_fit(rows)
    ninl = 0
    gp = (0, 1)
    cg = 0.0
    used_ok = False
    if fitg is not None and fitg[0][0] >= 12:
        (ninl, _), gp, thg, keepg = fitg
        a = np.asarray(thg[:5], float).copy()
        cg = float(thg[5])
        used_ok = True
        log.append("grad-sparse:%s ninl=%d/%d c=%.4f"
                   % (gp, ninl, len(rows), cg))
    else:
        if len(rows) >= 24:
            theta, keep, ninl = _robust_grad_fit(rows, nbase)
        else:
            theta = np.zeros(20)
            ninl = 0
        a = np.asarray(theta[:5], float).copy()
        B = np.zeros((5, 5))
        for i in range(5):
            for j in range(i, 5):
                B[i, j] = B[j, i] = float(theta[BCOL[(i, j)]])
        gp = max(PAIR_LIST, key=lambda ij: abs(B[ij[0], ij[1]]))
        cg = float(B[gp[0], gp[1]])
        log.append("grad-general:%d/%d" % (ninl, len(rows)))

    ia, ib = gp

    def smooth0(X):
        """Smooth part without the constant term."""
        X = np.atleast_2d(np.asarray(X, float))
        return X @ a + cg * X[:, ia] * X[:, ib]

    # ------------------------------------------------------------------
    # 3. label every reading through u = y - smooth(x)
    # ------------------------------------------------------------------
    Xall, yall = snapshot()
    u = yall - smooth0(Xall)
    sp = _split_two(u)

    if sp is None:
        # fallback: RANSAC a 7-parameter quadratic directly on the corners
        best = None
        rng = np.random.default_rng(12345)
        ok = np.isfinite(yc)
        Xc = corners[ok]
        yv = np.asarray(yc, float)[ok]
        for (i, j) in PAIR_LIST:
            A = np.column_stack([np.ones(len(Xc)), Xc,
                                 Xc[:, i] * Xc[:, j]])
            for _ in range(600):
                idx = rng.choice(len(Xc), 8, replace=False)
                try:
                    th, *_ = np.linalg.lstsq(A[idx], yv[idx], rcond=None)
                except np.linalg.LinAlgError:
                    continue
                res = np.abs(A @ th - yv)
                inl = int((res < 1e-4).sum())
                if best is None or inl > best[0]:
                    best = (inl, (i, j), th)
        if best is not None and best[0] >= 10:
            _, (i, j), th = best
            linr = np.asarray(th[1:6], float)
            cr = float(th[6])

            def smooth0r(X):
                X = np.atleast_2d(np.asarray(X, float))
                return X @ linr + cr * X[:, i] * X[:, j]

            ur = yall - smooth0r(Xall)
            spr = _split_two(ur)
            if spr is not None:
                smooth0 = smooth0r
                u = ur
                sp = spr
                log.append("fallback_ransac:%s" % ((i, j),))

    if sp is None:
        # last resort: plain quadratic, no step
        A = np.column_stack([np.ones(len(Xall)), Xall] +
                            [Xall[:, i] * Xall[:, j] for i, j in PAIR_LIST])
        th, *_ = np.linalg.lstsq(A, yall, rcond=None)
        b0 = float(th[0])
        lin = np.asarray(th[1:6], float)
        beta = np.zeros((5, 5))
        for k, (i, j) in enumerate(PAIR_LIST):
            beta[i, j] = beta[j, i] = float(th[6 + k])
        bi, bj = max(PAIR_LIST, key=lambda ij: abs(beta[ij[0], ij[1]]))
        pred = b0 + Xq @ lin + beta[bi, bj] * Xq[:, bi] * Xq[:, bj]
        return _assemble(Xq, pred, b0, lin, beta, np.zeros(5), bi, bj,
                         None, 0.0, log, "no step found")

    lo_v, hi_v, lab = sp
    log.append("levels:%.3f/%.3f n=%d/%d" %
               (lo_v, hi_v, int(lab.sum()), int((~lab).sum())))
    mid = 0.5 * (lo_v + hi_v)

    def classify(y, x):
        uu = float(y - smooth0(x)[0])
        return 1 if uu > mid else 0

    # ------------------------------------------------------------------
    # 4. boundary points by bisection
    # ------------------------------------------------------------------
    pos = np.nonzero(lab)[0]
    neg = np.nonzero(~lab)[0]
    bps = []
    if len(pos) >= 2 and len(neg) >= 2:
        rng2 = np.random.default_rng(20260101)
        pairs = []
        seen = set()
        guard = 0
        while len(pairs) < 16 and guard < 500:
            guard += 1
            qn = int(rng2.choice(neg))
            qp = int(rng2.choice(pos))
            if (qn, qp) in seen:
                continue
            seen.add((qn, qp))
            pairs.append((qn, qp))

        def bisect_pair(qn, qp):
            x_neg = Xall[qn]
            x_pos = Xall[qp]
            lo, hi = 0.0, 1.0
            for _ in range(11):
                if hi - lo < 2.0e-3:
                    break
                t = 0.5 * (lo + hi)
                xm = x_neg + t * (x_pos - x_neg)
                y = ask(xm, "bisect")
                if y is None:
                    break
                if classify(y, xm) == 0:
                    lo = t
                else:
                    hi = t
            return x_neg + 0.5 * (lo + hi) * (x_pos - x_neg)

        room = int(oracle.remaining)
        max_bis = min(9, max(4, (room - 8) // 9))
        for k, (qi, qj) in enumerate(pairs):
            if len(bps) >= max_bis:
                break
            if k >= 6:
                # stop early unless the crossings span the full 4-D hyperplane
                Xb = np.asarray(bps, float)
                sv = np.linalg.svd(Xb - Xb.mean(axis=0), compute_uv=False)
                if len(sv) >= 4 and sv[3] > 0.06 * sv[0]:
                    break
            bps.append(bisect_pair(qi, qj))
    log.append("boundary_points:%d" % len(bps))

    # ------------------------------------------------------------------
    # 5. switching hyper-plane: candidate variable supports are scored both
    #    by how exactly the whole device is reproduced and by how exactly the
    #    bisected crossings lie on the candidate plane (the second test picks
    #    up gate dials with a small but real weight).
    # ------------------------------------------------------------------
    Xall, yall = snapshot()          # refresh (bisections added readings)

    def gate_candidate(Xb, supp):
        Y = Xb[:, supp]
        m = Y.mean(axis=0)
        if len(supp) == 1:
            ns = np.array([1.0])
        else:
            try:
                _, _, V2 = np.linalg.svd(Y - m)
                ns = V2[-1]
            except np.linalg.LinAlgError:
                ns = np.ones(len(supp))
        Ts = float(ns @ m)
        sv = Xall[:, supp] @ ns - Ts
        if float(sv[pos].mean()) < 0:
            ns, Ts = -ns, -Ts
            sv = -sv
        bad = int(np.sum((sv > 0) != lab))
        if bad > 0:
            lo_s = float(sv[neg].max())
            hi_s = float(sv[pos].min())
            if lo_s < hi_s:
                Ts = 0.5 * (lo_s + hi_s)
                sv = Xall[:, supp] @ ns - Ts
                bad = int(np.sum((sv > 0) != lab))
        # how well the bisected crossings themselves lie on this plane
        hres = float(np.sqrt(np.mean((Y @ ns - Ts) ** 2))
                     / (np.linalg.norm(ns) + 1e-12))
        g = (Xall[:, supp] @ ns >= Ts).astype(float)
        dist = np.abs(Xall[:, supp] @ ns - Ts) / (np.linalg.norm(ns) + 1e-12)
        keep = dist > 0.01
        if keep.sum() < 12:
            keep = np.ones(len(Xall), bool)
        A = design(Xall, g)
        th, *_ = np.linalg.lstsq(A[keep], yall[keep], rcond=None)
        r = A @ th - yall
        rms = float(np.sqrt(np.mean(r[keep] ** 2)))
        return {"supp": supp, "n": ns, "T": Ts, "bad": bad, "rms": rms,
                "hres": hres, "keep": keep, "g": g}

    gate = None
    if len(bps) >= 2:
        Xb = np.asarray(bps, float)
        from itertools import combinations
        cands = [list(c) for k in (1, 2, 3, 4)
                 for c in combinations(range(5), k)]
        res = [gate_candidate(Xb, c) for c in cands]
        min_rms = min(d["rms"] for d in res)
        min_h = min(d["hres"] for d in res)
        ok = [d for d in res
              if d["rms"] <= max(1e-3, 3.0 * min_rms)
              and d["hres"] <= max(2.5e-3, 1.5 * min_h)]
        if not ok:
            ok = [d for d in res if d["rms"] <= max(2e-2, 5.0 * min_rms)]
        if not ok:
            ok = res
        gate = min(ok, key=lambda d: (len(d["supp"]), round(d["rms"], 8),
                                      d["hres"]))
        log.append("gate:%s bad=%d rms=%.1e hres=%.1e n=%d" %
                   ([VARS[i] for i in gate["supp"]], gate["bad"],
                    gate["rms"], gate["hres"], len(Xb)))
        if os.environ.get("BB_DEBUG"):
            for d in sorted(res, key=lambda d: (round(d["rms"], 6),
                                                d["hres"]))[:10]:
                print("DBG cand", d["supp"], "rms", round(d["rms"], 6),
                      "hres", round(d["hres"], 5), "bad", d["bad"],
                      file=sys.stderr)

    if gate is not None:
        supp = gate["supp"]
        ns = np.asarray(gate["n"], float)
        Ts = float(gate["T"])
        g_all = gate["g"]
        keep_rows = gate["keep"]
    else:
        supp = [0]
        ns = np.array([1.0])
        Ts = 0.0
        g_all = np.zeros(len(Xall))
        keep_rows = np.ones(len(Xall), bool)

    # ------------------------------------------------------------------
    # 6. exact global least squares
    # ------------------------------------------------------------------
    def fit_full(g):
        A = design(Xall, g)
        th, *_ = np.linalg.lstsq(A[keep_rows], yall[keep_rows], rcond=None)
        res = A @ th - yall
        return th, float(np.sqrt(np.mean(res[keep_rows] ** 2)))

    th, rms = fit_full(g_all)
    if gate is not None and rms > 3e-3:
        th2, rms2 = fit_full(1.0 - g_all)
        if rms2 < rms:
            th, rms = th2, rms2
            ns, Ts = -ns, -Ts
            g_all = 1.0 - g_all
            gate["n"], gate["T"] = ns, Ts
            log.append("flipped_gate")

    b0 = float(th[0])
    lin = np.asarray(th[1:6], float)
    beta = np.zeros((5, 5))
    for k, (i, j) in enumerate(PAIR_LIST):
        beta[i, j] = beta[j, i] = float(th[6 + k])
    sq = np.asarray(th[16:21], float)
    J = float(th[21]) if gate is not None else 0.0

    bi, bj = max(PAIR_LIST, key=lambda ij: abs(beta[ij[0], ij[1]]))
    bcoef = float(beta[bi, bj])
    use_sq = sq.copy()
    if np.max(np.abs(sq)) < 1e-3:
        use_sq[:] = 0.0

    # refit with the selected structure only
    def design_sel(X, g, bcoef_=None, use_sq_=None):
        bcoef_ = bcoef if bcoef_ is None else bcoef_
        use_sq_ = use_sq if use_sq_ is None else use_sq_
        cols = [np.ones(len(X))[:, None], X]
        if abs(bcoef_) > 1e-6:
            cols.append((X[:, bi] * X[:, bj])[:, None])
        for i in range(5):
            if use_sq_[i] != 0.0:
                cols.append((X[:, i] ** 2)[:, None])
        if gate is not None:
            cols.append(g[:, None])
        return np.column_stack(cols)

    def fit_sel(g, bcoef_=None, use_sq_=None):
        A = design_sel(Xall, g, bcoef_, use_sq_)
        th, *_ = np.linalg.lstsq(A[keep_rows], yall[keep_rows], rcond=None)
        res = A @ th - yall
        return th, float(np.sqrt(np.mean(res[keep_rows] ** 2)))

    if gate is not None:
        ths, rms_s = fit_sel(g_all)
        if rms_s <= rms:
            b0 = float(ths[0])
            lin = np.asarray(ths[1:6], float)
            k = 6
            if abs(bcoef) > 1e-6:
                bcoef = float(ths[k])
                k += 1
            use_sq2 = np.zeros(5)
            for i in range(5):
                if use_sq[i] != 0.0:
                    use_sq2[i] = float(ths[k])
                    k += 1
            use_sq = use_sq2
            J = float(ths[k])
            rms = rms_s
    log.append("fit_rmse:%.1e" % rms)

    # ------------------------------------------------------------------
    # 7. predictions and submission
    # ------------------------------------------------------------------
    pred = b0 + Xq @ lin
    if abs(bcoef) > 1e-6:
        pred = pred + bcoef * Xq[:, bi] * Xq[:, bj]
    for i in range(5):
        if use_sq[i] != 0.0:
            pred = pred + use_sq[i] * Xq[:, i] ** 2
    if gate is not None:
        pred = pred + J * (Xq[:, supp] @ ns >= Ts).astype(float)

    beta[bi, bj] = beta[bj, bi] = bcoef
    gate_out = None
    if gate is not None:
        gate_out = {"supp": supp, "n": ns, "T": Ts}
    return _assemble(Xq, pred, b0, lin, beta, use_sq, bi, bj, gate_out,
                     J, log, "")


# ----------------------------------------------------------------------------
def _assemble(Xq, pred, b0, lin, beta, sq, bi, bj, gate, J, log, note):
    has_int = abs(float(beta[bi, bj])) > 1e-6
    gate_vars = []
    coeffs_f = None
    tau_f = 0.0
    disc = []
    if gate is not None:
        supp = gate["supp"]
        ns = np.asarray(gate["n"], float)
        Ts = float(gate["T"])
        coeffs = np.zeros(5)
        for k, i in enumerate(supp):
            coeffs[i] = float(ns[k])
        scale = float(np.max(np.abs(coeffs)))
        if scale > 0:
            coeffs = coeffs / scale
            tau_f = Ts / scale
            coeffs_f = coeffs
            gate_vars = [VARS[i] for i in range(5) if abs(coeffs[i]) > 1e-9]
            boundary = {"form": "linear",
                        "coefficients": {VARS[i]: float(coeffs[i])
                                         for i in range(5)},
                        "threshold": float(tau_f),
                        "active_when": "greater_equal"}
            cond = " + ".join(f"({float(coeffs[i])!r})*{VARS[i]}"
                              for i in range(5) if coeffs[i] != 0)
            disc = [{"gate_variables": gate_vars,
                     "boundary": boundary,
                     "jump": float(abs(J)),
                     "condition": f"{cond} >= {float(tau_f)!r}"}]

    rel = []
    for i in range(5):
        if abs(lin[i]) > 1e-4:
            rel.append(VARS[i])
        elif has_int and i in (bi, bj):
            rel.append(VARS[i])
        elif VARS[i] in gate_vars:
            rel.append(VARS[i])
    irr = [v for v in VARS if v not in rel]

    terms = [repr(float(b0))]
    for i in range(5):
        if abs(lin[i]) > 1e-6:
            terms.append(f"({float(lin[i])!r})*x{i + 1}")
    if has_int:
        terms.append(f"({float(beta[bi, bj])!r})*x{bi + 1}*x{bj + 1}")
    for i in range(5):
        if abs(float(sq[i])) > 1e-6:
            terms.append(f"({float(sq[i])!r})*x{i + 1}**2")
    expr = " + ".join(terms)
    if coeffs_f is not None and abs(J) > 1e-9:
        cond = " + ".join(f"({float(coeffs_f[i])!r})*{VARS[i]}"
                          for i in range(5) if coeffs_f[i] != 0)
        expr = expr + f" + ({float(J)!r} if {cond} >= {float(tau_f)!r} else 0.0)"

    notes = ("32 corners guarantee both sides of the step; 6x6 finite-"
             "difference gradients give the full quadratic; residuals are "
             "two-valued, giving free inside/outside labels; 6 bisections pin "
             "the switching hyper-plane; one exact least squares sets all "
             "coefficients. " + " | ".join(log)
             + (" " + note if note else ""))

    return {
        "relevant_variables": rel,
        "irrelevant_variables": irr,
        "interactions": ([{"variables": [VARS[bi], VARS[bj]],
                           "type": "product",
                           "coefficient": float(beta[bi, bj]),
                           "equivalent_centered_form":
                               f"{float(beta[bi, bj])!r}*({VARS[bi]}-0.5)*"
                               f"({VARS[bj]}-0.5)"}]
                         if has_int else []),
        "discontinuities": disc,
        "model_expression": expr,
        "predictions": [float(v) for v in pred],
        "notes": notes,
    }
