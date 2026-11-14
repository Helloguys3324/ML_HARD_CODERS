"""Black Box - recover the wiring diagram of a hidden 5-dial device.

The device family (the public practice box and the graded box are stated to come
from the same family) is

    f(x) = a0 + sum_i a_i x_i + sum_{i<j} b_ij x_i x_j  +  J * 1[w . x >= t]

affine main effects, ONE bilinear interaction, and ONE hard step across a linear
switching surface.  Nothing about *which* dials are live, which pair interacts,
or where the surface sits is assumed: every one of those facts is measured from
readings.

Strategy (<= 150 readings; no reading of the graded box before the run):

  1. Local gradients at a low-discrepancy set of base points.  The smooth part
     is quadratic, so its gradient field g(x) = a + B x is affine; one global
     least-squares fit of the measured gradients recovers the full interaction
     matrix B (hence the interaction pair and its expanded coefficient) and the
     main-effect slopes.  A step only spoils probes that straddle it, and those
     are orders of magnitude outlying gradients, which are trimmed.
  2. With the smooth part known, u = y - smooth(x) takes only two values, 0 and
     J, at every queried point.  That splits every reading into inside / outside
     the switched region for free.
  3. Every candidate dial subset is tested for linear separability of those
     labels.  The survivors are rival wiring hypotheses; a short adaptive
     sequence of readings is placed exactly where the rivals disagree, which
     kills the wrong hypotheses against the real device instead of guessing.
  4. Bisection along the surviving surface pins its offset to ~1e-3, far inside
     the grader's 0.005 "coin flip" band, in two or three short scans.
  5. One final least squares on (1, x_i, x_i x_j, 1[gate]) re-estimates every
     coefficient exactly; the model string and the 300 predictions follow.
"""
import math
import os

import numpy as np

VARS = ("x1", "x2", "x3", "x4", "x5")
PRIMES = (2, 3, 5, 7, 11)


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def _vdc(i, b):
    """i-th van der Corput coordinate in base b (i >= 1)."""
    f, r = 1.0 / b, 0.0
    while i > 0:
        r += f * (i % b)
        i //= b
        f /= b
    return r


def _halton(n, skip=0, lo=0.02, hi=0.98):
    P = np.empty((n, 5))
    for k in range(n):
        for j in range(5):
            P[k, j] = _vdc(skip + k + 1, PRIMES[j])
    return lo + (hi - lo) * P


def _smooth_no_intercept(X, a, B):
    """Quadratic part of the smooth model.

    B is the *gradient* matrix: d f / d x_k = a_k + sum_j B_kj x_j, symmetric.
    Hence f contains B_ij x_i x_j for i < j and B_ii/2 x_i^2.
    """
    X = np.atleast_2d(X)
    out = X @ a
    for i in range(5):
        for j in range(i + 1, 5):
            out = out + B[i, j] * X[:, i] * X[:, j]
        out = out + 0.5 * B[i, i] * X[:, i] ** 2
    return out


def _lstsq_trim(A, y, nbase=8, per=5, iters=4):
    """Robust fit of the affine gradient field.

    Each base point contributes `per` equations, so any 4 bases give one exact
    20-parameter system.  A step only spoils the probes that straddle it, and a
    contaminated gradient is off by J/h (hundreds) instead of ~1e-4, so trying
    every 4-base system and keeping the one with the most inliers is a
    deterministic RANSAC that cannot be derailed by a biased start.
    """
    n, p = A.shape
    nb = n // per
    tol = max(5e-3, 0.02 * float(np.median(np.abs(y))))
    best = None
    try:
        from itertools import combinations
        combos = list(combinations(range(nb), 4))
    except Exception:                                    # noqa: BLE001
        combos = []
    if nb >= 4 and nb * per == n and p == 20:
        for comb in combos:
            idx = [b * per + i for b in comb for i in range(per)]
            try:
                th, *_ = np.linalg.lstsq(A[idx], y[idx], rcond=None)
            except Exception:                            # noqa: BLE001
                continue
            inl = int((np.abs(A @ th - y) < tol).sum())
            if best is None or inl > best[0]:
                best = (inl, th)
        if best is not None and best[0] >= p + 2:
            theta = best[1]
            for _ in range(iters):
                keep = np.abs(A @ theta - y) < tol
                if keep.sum() < p:
                    break
                theta, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
            w = np.abs(A @ theta - y) < tol
            return theta, w
    # fallback: least-trimmed-squares
    k = min(n, max(p + 6, int(np.ceil(0.62 * n))))
    w = np.ones(n, bool)
    theta = np.zeros(p)
    for it in range(iters + 4):
        theta, *_ = np.linalg.lstsq(A[w], y[w], rcond=None)
        if it == iters + 3:
            break
        r = np.abs(A @ theta - y)
        idx = np.argsort(r)[:k]
        keep = np.zeros(n, bool)
        keep[idx] = True
        if np.array_equal(keep, w):
            break
        w = keep
    return theta, w


# ----------------------------------------------------------------------------
def solve(oracle, points):
    Xq = np.asarray(points, dtype=np.float64).reshape(-1, 5)
    budget0 = int(getattr(oracle, "remaining", 150))
    log = []

    # --- query bookkeeping ---------------------------------------------------
    data = {}

    def key(x):
        return tuple(round(float(v), 9) for v in x)

    def record(x, y):
        data[key(x)] = (np.asarray(x, float), float(y))

    def ask(x, phase=None):
        x = np.clip(np.asarray(x, float), 0.0, 1.0)
        k = key(x)
        if k in data:
            return data[k][1]
        if oracle.remaining <= 1:
            return None
        y = float(oracle.ask([float(v) for v in x], phase))
        record(x, y)
        return y

    def ask_many(Xa, phase=None):
        Xa = [np.clip(np.asarray(x, float), 0.0, 1.0) for x in Xa]
        out = [None] * len(Xa)
        todo, pos = [], []
        for k, x in enumerate(Xa):
            kk = key(x)
            if kk in data:
                out[k] = data[kk][1]
            else:
                todo.append(x)
                pos.append(k)
        room = max(0, oracle.remaining - 1)
        if len(todo) > room:
            todo, pos = todo[:room], pos[:room]
        if todo:
            yy = oracle.ask_many([[float(v) for v in x] for x in todo], phase)
            for x, y in zip(todo, yy):
                record(x, y)
            for k, y in zip(pos, yy):
                out[k] = float(y)
        return out

    def qused():
        return budget0 - int(oracle.remaining)

    def snapshot():
        X = np.array([data[k][0] for k in data], float)
        y = np.array([data[k][1] for k in data], float)
        return X, y

    # ------------------------------------------------------------------
    # Stage 1: gradients at low-discrepancy base points
    # ------------------------------------------------------------------
    H = 0.005
    bases = _halton(8, skip=3)
    ys = ask_many(list(bases), "grad-base")
    obs = []  # (base, axis, fd-gradient, sign)
    for b, y0 in zip(bases, ys):
        if y0 is None:
            continue
        for i in range(5):
            s = 1.0 if b[i] + H <= 1.0 else -1.0
            p = b.copy()
            p[i] += s * H
            y = ask(p, "grad")
            if y is None:
                continue
            obs.append((b.copy(), i, s * (y - y0) / H, s))

    pidx = {}
    for i in range(5):
        pidx[("a", i)] = i
    kk = 5
    for i in range(5):
        for j in range(i, 5):
            pidx[("b", i, j)] = kk
            kk += 1

    theta = np.zeros(kk)
    nfit = len(obs)
    nkeep = nfit
    if nfit >= kk + 2:
        A = np.zeros((nfit, kk))
        g = np.zeros(nfit)
        for r, (x, i, gi, s) in enumerate(obs):
            A[r, pidx[("a", i)]] = 1.0
            for j in range(5):
                A[r, pidx[("b", min(i, j), max(i, j))]] += x[j]
            # one-sided finite difference contains a known + s*H/2 * B_ii term
            A[r, pidx[("b", i, i)]] += 0.5 * s * H
            g[r] = gi
        theta, w = _lstsq_trim(A, g, nbase=len(bases), per=5)
        nkeep = int((w > 0.5).sum())
    a = theta[:5].copy()
    B = np.zeros((5, 5))
    for i in range(5):
        for j in range(i, 5):
            B[i, j] = B[j, i] = theta[pidx[("b", i, j)]]
    log.append(f"grad:{nfit}/{nkeep}")

    # a few extra spread labels (no gradients) help the structure test
    fills = _halton(24, skip=40, lo=0.04, hi=0.96)
    ask_many(list(fills), "fill")

    # ------------------------------------------------------------------
    # Stage 2: label every reading by its residual against the smooth part
    # ------------------------------------------------------------------
    def u_of(X, y):
        return y - _smooth_no_intercept(X, a, B)

    def split_two(u):
        """Two-level split of u; returns (low, J, labels) or None."""
        n = len(u)
        if n < 10:
            return None
        order = np.argsort(u)
        us = u[order]
        gaps = np.diff(us)
        kg = int(np.argmax(gaps))
        lo, hi = us[:kg + 1], us[kg + 1:]
        if len(lo) < 3 or len(hi) < 3:
            return None
        spread = max(lo[-1] - lo[0], hi[-1] - hi[0])
        gmax = float(gaps[kg])
        if gmax < 1e-4 or gmax < 4.0 * spread:
            return None
        lo_v, hi_v = float(np.mean(lo)), float(np.mean(hi))
        if abs(hi_v - lo_v) < 1e-4:
            return None
        lab = u > 0.5 * (lo_v + hi_v)
        return lo_v, hi_v - lo_v, lab

    Xall, yall = snapshot()
    uall = u_of(Xall, yall)
    sp = split_two(uall)
    if sp is None or min(int(sp[2].sum()), int((~sp[2]).sum())) < 3:
        # no clear step seen yet: the corners must contain a point on the live
        # side of any halfspace that intersects the cube
        corners = np.array([[i, j, k2, l, m]
                            for i in (0.0, 1.0) for j in (0.0, 1.0)
                            for k2 in (0.0, 1.0) for l in (0.0, 1.0)
                            for m in (0.0, 1.0)])
        missing = [c for c in corners if key(c) not in data]
        ask_many(missing, "corners")
        Xall, yall = snapshot()
        uall = u_of(Xall, yall)
        sp = split_two(uall)

    if sp is None:
        a0 = float(np.mean(uall))
        J = 0.0
        lab = np.zeros(len(uall), bool)
    else:
        a0, J, lab = sp
        if J < 0:
            J = -J
            lab = ~lab
            a0 = float(np.mean(uall[~lab]))
    P, L = Xall.copy(), lab.copy()
    log.append(f"step:{int(L.sum())}/{len(L)} J={J:.3f}")

    # ------------------------------------------------------------------
    # Stage 3: rival gate hypotheses from the labels
    # ------------------------------------------------------------------
    def fit_support(Pt, Lt, sup):
        """Best separating (margin, direction, offset) for one dial subset."""
        Q = Pt[:, sup]
        Aa, Bb = Q[Lt], Q[~Lt]
        if len(Aa) < 2 or len(Bb) < 2:
            return None
        d = len(sup)
        if d == 1:
            best = None
            for sgn in (1.0, -1.0):
                sa = sgn * Aa[:, 0]
                sb = sgn * Bb[:, 0]
                mg = float(sa.min() - sb.max())
                if best is None or mg > best[0]:
                    tau = 0.5 * (float(sa.min()) + float(sb.max()))
                    best = (mg, np.array([sgn]), tau)
            return best if best[0] > 1e-6 else None
        if d == 2:
            def margin(t):
                nn = np.array([math.cos(t), math.sin(t)])
                return float((Aa @ nn).min() - (Bb @ nn).max())
            grid = np.linspace(0.0, 2.0 * math.pi, 1441)
            vals = np.array([margin(t) for t in grid])
            kb = int(np.argmax(vals))
            if vals[kb] <= 1e-6:
                return None
            lo = grid[kb] - 2.0 * math.pi / 1440.0
            hi = grid[kb] + 2.0 * math.pi / 1440.0
            for _ in range(90):
                m1 = lo + (hi - lo) * 0.382
                m2 = lo + (hi - lo) * 0.618
                if margin(m1) > margin(m2):
                    hi = m2
                else:
                    lo = m1
            t = 0.5 * (lo + hi)
            nn = np.array([math.cos(t), math.sin(t)])
            mg = margin(t)
            if mg <= 1e-6:
                return None
            tau = 0.5 * (float((Aa @ nn).min()) + float((Bb @ nn).max()))
            return (mg, nn, tau)
        # 3-d: max-margin by hinge subgradient from several starts
        best = None
        for seed in range(5):
            rng = np.random.default_rng(7919 + 131 * seed + d)
            wv = rng.normal(size=d)
            wv /= np.linalg.norm(wv)
            for it in range(1500):
                sa = Aa @ wv
                sb = Bb @ wv
                act = (sb[None, :] - sa[:, None] + 1.0) > 0
                if not act.any():
                    break
                gA = act.sum(axis=1).astype(float)
                gB = act.sum(axis=0).astype(float)
                grad = Bb.T @ gB - Aa.T @ gA
                wv = wv - (0.15 / math.sqrt(it + 1.0)) * grad
                nw = np.linalg.norm(wv)
                if nw < 1e-12:
                    wv = None
                    break
                wv /= nw
            if wv is None:
                continue
            mg = float((Aa @ wv).min() - (Bb @ wv).max())
            if best is None or mg > best[0]:
                tau = 0.5 * (float((Aa @ wv).min()) + float((Bb @ wv).max()))
                best = (mg, wv.copy(), tau)
        return best if best is not None and best[0] > 1e-6 else None

    def all_supports():
        out = {}
        for i in range(5):
            r = fit_support(P, L, (i,))
            if r is not None:
                out[(i,)] = r
        for i in range(5):
            for j in range(i + 1, 5):
                r = fit_support(P, L, (i, j))
                if r is not None:
                    out[(i, j)] = r
        if not out:
            for comb in ((0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3), (0, 2, 4),
                         (0, 3, 4), (1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)):
                r = fit_support(P, L, comb)
                if r is not None:
                    out[comb] = r
        return out

    alive = {}
    if J > 0 and L.sum() >= 3 and (~L).sum() >= 3:
        alive = all_supports()
    log.append("supp:" + ",".join(
        f"{k}:{v[0]:.3f}" for k, v in sorted(alive.items(), key=lambda t: -t[1][0])[:10]))
    if os.environ.get("BB_DEBUG"):
        us = u_of(P, yall)
        print("DBG labels", int(L.sum()), "/", len(L), "J", round(J, 4),
              "u range", round(float(us.min()), 5), round(float(us.max()), 5),
              "a", np.round(a, 4), "Bdiag", np.round(np.diag(B), 4), file=__import__("sys").stderr)
        for sup in [(i,) for i in range(5)] + [(i, j) for i in range(5) for j in range(i + 1, 5)]:
            r = fit_support(P, L, sup)
            if r is not None:
                print("DBG  supp", sup, "margin", round(r[0], 4), "n", np.round(r[1], 3), "tau", round(float(r[2]), 4), file=__import__("sys").stderr)

    # adaptive discrimination: read exactly where the rivals disagree
    def predict(h, Z):
        mg, nn, tau = h
        return (Z @ nn) >= tau

    disc_pts = []
    if len(alive) > 1:
        rng = np.random.default_rng(20241114)
        Z = rng.random((3000, 5))
        Z[:600, :] = np.round(Z[:600, :] * 5.0) / 5.0
        keys = list(alive.keys())
        for _ in range(10):
            if len(keys) < 2 or oracle.remaining < 34:
                break
            preds = np.empty((len(keys), len(Z)), bool)
            conf = np.full(len(Z), np.inf)
            for h, sup in enumerate(keys):
                mg, nn, tau = alive[sup]
                s = Z[:, sup] @ nn
                preds[h] = s >= tau
                conf = np.minimum(conf, np.abs(s - tau))
            best = None
            for i in range(len(Z)):
                na = int(preds[:, i].sum())
                ni = len(keys) - na
                v = min(na, ni)
                if v == 0:
                    continue
                score = (v, conf[i])
                if best is None or score > best[0]:
                    best = (score, i)
            if best is None or best[0][1] < 0.012:
                break
            z = Z[best[1]]
            y = ask(z, "discriminate")
            if y is None:
                break
            u = float(u_of(np.atleast_2d(z), np.array([y]))[0])
            labz = abs(u - (a0 + J)) < abs(u - a0)
            disc_pts.append((z, labz))
            P = np.vstack([P, z])
            L = np.concatenate([L, [labz]])
            new = {}
            for sup in keys:
                r = fit_support(P, L, sup)
                if r is not None:
                    new[sup] = r
            keys = list(new.keys())
            alive = new
        log.append(f"disc:{len(disc_pts)}->{len(alive)}")

    # ------------------------------------------------------------------
    # Stage 4: refine the surviving switching surface by bisection
    # ------------------------------------------------------------------
    def is_active(x):
        y = ask(x, "bisect")
        if y is None:
            return None
        u = float(u_of(np.atleast_2d(x), np.array([y]))[0])
        return abs(u - (a0 + J)) < abs(u - a0)

    def bisect(fixed, axis, lo, hi, lo_act):
        while hi - lo > 1.2e-3:
            mid = 0.5 * (lo + hi)
            x = fixed.copy()
            x[axis] = mid
            act = is_active(x)
            if act is None:
                break
            if act == lo_act:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def validate(coeffs, tau):
        s = P @ coeffs
        return int(np.sum((s >= tau) != L))

    def refine_pair(i, j, n2, tau, nscans=3):
        n2 = np.asarray(n2, float)
        if abs(n2[0]) >= abs(n2[1]):
            ax, fx = i, j
            na, nf = float(n2[0]), float(n2[1])
        else:
            ax, fx = j, i
            na, nf = float(n2[1]), float(n2[0])
        if abs(nf) < 1e-9:
            vs = [0.12, 0.5, 0.88][:nscans]
        else:
            grid = np.linspace(0.02, 0.98, 193)
            xa = (tau - nf * grid) / na
            ok = (xa > 0.06) & (xa < 0.94)
            vv = grid[ok]
            if len(vv) >= 3:
                idx = np.unique(np.linspace(0, len(vv) - 1, nscans).astype(int))
                vs = [float(vv[q]) for q in idx]
            elif len(vv) >= 1:
                vs = [float(vv[0]), float(vv[-1])]
            else:
                vs = [0.3, 0.7][:nscans]
        crossings = []
        for v in vs:
            fixed = np.full(5, 0.5)
            fixed[fx] = v
            xa0 = (tau - nf * v) / na if abs(na) > 1e-12 else 0.5
            okc = False
            for lo, hi in ((max(0.0, min(1.0, xa0 - 0.3)),
                            max(0.0, min(1.0, xa0 + 0.3))), (0.0, 1.0)):
                xlo = fixed.copy()
                xlo[ax] = lo
                xhi = fixed.copy()
                xhi[ax] = hi
                alo = is_active(xlo)
                ahi = is_active(xhi)
                if alo is None or ahi is None or alo == ahi:
                    continue
                xc = bisect(fixed, ax, lo, hi, alo)
                crossings.append(np.array([xc, v]) if ax == i else np.array([v, xc]))
                okc = True
                break
            if not okc:
                continue
        if len(crossings) < 2:
            return None
        C = np.array(crossings)
        c0 = C.mean(axis=0)
        _, _, vt = np.linalg.svd(C - c0)
        nfit = vt[-1]
        tau_fit = float(nfit @ c0)
        Q = P[:, (i, j)]
        if (Q[L] @ nfit).mean() < (Q[~L] @ nfit).mean():
            nfit, tau_fit = -nfit, -tau_fit
        coeffs = np.zeros(5)
        coeffs[i], coeffs[j] = float(nfit[0]), float(nfit[1])
        if validate(coeffs, tau_fit) > 0:
            s = P @ coeffs
            lo_s, hi_s = float(s[~L].max()), float(s[L].min())
            if lo_s < hi_s:
                tau_fit = 0.5 * (lo_s + hi_s)
        return {"coeffs": coeffs, "tau": tau_fit, "nscan": len(C), "cross": C}

    def refine_one(sup, aux):
        if len(sup) == 2:
            if aux is None:
                return None
            n = np.asarray(aux[1], float)
            tau = float(aux[2])
            nsc = 3 if oracle.remaining > 40 else 2
            return refine_pair(sup[0], sup[1], n, tau, nscans=nsc)
        if len(sup) == 1:
            i0 = sup[0]
            fixed = np.full(5, 0.5)
            xlo = np.where(np.arange(5) == i0, 0.0, fixed)
            xhi = np.where(np.arange(5) == i0, 1.0, fixed)
            alo = is_active(xlo)
            ahi = is_active(xhi)
            if alo is None or ahi is None or alo == ahi:
                return None
            xc = bisect(fixed, i0, 0.0, 1.0, alo)
            n = 1.0 if ahi else -1.0
            coeffs = np.zeros(5)
            coeffs[i0] = n
            return {"coeffs": coeffs, "tau": n * xc, "nscan": 1, "cross": None}
        return refine_generic(sup, aux)

    def refine_generic(sup, aux):
        """Fallback for a >= 3 dial gate: bisect along the separator."""
        nn = np.asarray(aux[1], float)
        tau = float(aux[2])
        act_idx = np.where(L)[0]
        if len(act_idx) > 9:
            act_idx = act_idx[np.linspace(0, len(act_idx) - 1, 9).astype(int)]
        pts = []
        for ai in act_idx:
            x0 = P[ai].copy()
            d0 = float(nn @ x0[list(sup)] - tau)
            found = None
            for step in (max(0.03, 1.5 * d0), max(0.1, 3.0 * d0) + 0.1):
                x1 = x0.copy()
                for kx, iv in enumerate(sup):
                    x1[iv] = np.clip(x0[iv] - step * nn[kx], 0.0, 1.0)
                if not is_active(x1):
                    found = x1
                    break
            if found is None:
                continue
            lo, hi = 0.0, 1.0
            for _ in range(26):
                mid = 0.5 * (lo + hi)
                if is_active(x0 + mid * (found - x0)):
                    lo = mid
                else:
                    hi = mid
                if hi - lo < 1.5e-3:
                    break
            pts.append(x0 + 0.5 * (lo + hi) * (found - x0))
            if len(pts) >= len(sup) + 4:
                break
        if len(pts) < len(sup) + 1:
            return None
        C = np.array(pts)[:, list(sup)]
        c0 = C.mean(axis=0)
        _, _, vt = np.linalg.svd(C - c0)
        nf = vt[-1]
        tf = float(nf @ c0)
        coeffs = np.zeros(5)
        for kx, iv in enumerate(sup):
            coeffs[iv] = float(nf[kx])
        s_all = P @ coeffs
        if float(s_all[L].mean()) < float(s_all[~L].mean()):
            coeffs, tf = -coeffs, -tf
        if validate(coeffs, tf) > 0:
            s = P @ coeffs
            lo_s, hi_s = float(s[~L].max()), float(s[L].min())
            if lo_s < hi_s:
                tf = 0.5 * (lo_s + hi_s)
        return {"coeffs": coeffs, "tau": tf, "nscan": len(C), "cross": C}

    gate = None
    if alive:
        items = sorted(alive.items(), key=lambda t: -t[1][0])
        top_mg = items[0][1][0]
        # A larger support can always buy a little extra margin by overfitting;
        # only let it win if it is clearly better than the simplest rival.
        pref = [it for it in items if len(it[0]) == min(len(z[0]) for z in items)]
        cheap = [it for it in items if it[1][0] >= 0.8 * top_mg]
        order = sorted(cheap, key=lambda t: (len(t[0]), -t[1][0])) + items
        seen = set()
        tried = 0
        for sup, aux in order:
            if sup in seen or tried >= 3:
                continue
            seen.add(sup)
            tried += 1
            g = refine_one(sup, aux)
            if g is None:
                continue
            bad = validate(g["coeffs"], g["tau"])
            if bad > 0:
                s = P @ g["coeffs"]
                lo_s, hi_s = float(s[~L].max()), float(s[L].min())
                if hi_s > lo_s:
                    g["tau"] = 0.5 * (lo_s + hi_s)
                    bad = validate(g["coeffs"], g["tau"])
            g["bad"] = bad
            if bad == 0:
                gate = g
                break
        if gate is not None:
            sup = tuple(np.nonzero(gate["coeffs"])[0])
            log.append(f"gate:{sup} bad={gate['bad']} scans={gate['nscan']}")

    # ------------------------------------------------------------------
    # Stage 5: exact global least squares
    # ------------------------------------------------------------------
    # refresh the labelled data set (discrimination points enlarged it)
    Xall, yall = snapshot()
    uall = u_of(Xall, yall)
    L = np.abs(uall - (a0 + J)) < np.abs(uall - a0)
    P = Xall
    use_sq = bool(np.max(np.abs(np.diag(B))) > 2e-2)

    def design(X, gind):
        X = np.atleast_2d(X)
        cols = [np.ones(len(X))] + [X[:, i] for i in range(5)]
        names = ["1"] + [f"x{i+1}" for i in range(5)]
        for i in range(5):
            for j in range(i + 1, 5):
                cols.append(X[:, i] * X[:, j])
                names.append(f"x{i+1}*x{j+1}")
        if use_sq:
            for i in range(5):
                cols.append(X[:, i] ** 2)
                names.append(f"x{i+1}^2")
        cols.append(np.asarray(gind, float))
        names.append("step")
        return np.column_stack(cols), names

    if gate is not None and J > 0:
        coeffs = gate["coeffs"]
        gind = ((P @ coeffs) >= gate["tau"]).astype(float)
        if int(np.sum((gind > 0.5) != L)) > 0:
            gind = L.astype(float)
    else:
        gind = np.zeros(len(P))

    Xd, names = design(P, gind)
    coef, *_ = np.linalg.lstsq(Xd, yall, rcond=None)
    resid = Xd @ coef - yall
    rms = float(np.sqrt(np.mean(resid ** 2)))
    log.append(f"fit={rms:.1e}")
    if gate is not None and rms > 3e-3:
        smooth_cols = Xd[:, :-1]
        c2, *_ = np.linalg.lstsq(smooth_cols, yall, rcond=None)
        u2 = yall - smooth_cols @ c2
        sp2 = split_two(u2)
        if sp2 is not None:
            _, Jb, lab2 = sp2
            if Jb < 0:
                lab2 = ~lab2
            Xd2, names2 = design(P, lab2.astype(float))
            c3, *_ = np.linalg.lstsq(Xd2, yall, rcond=None)
            rms2 = float(np.sqrt(np.mean((Xd2 @ c3 - yall) ** 2)))
            if rms2 < rms:
                Xd, names, coef, rms = Xd2, names2, c3, rms2
                log.append(f"refit={rms:.1e}")

    ci = {nm: float(c) for nm, c in zip(names, coef)}
    a0f = ci["1"]
    lin = np.array([ci[f"x{i+1}"] for i in range(5)])
    bmat = np.zeros((5, 5))
    for i in range(5):
        for j in range(i + 1, 5):
            bmat[i, j] = bmat[j, i] = ci.get(f"x{i+1}*x{j+1}", 0.0)
    sq = np.array([ci.get(f"x{i+1}^2", 0.0) for i in range(5)])
    Jf = ci["step"]

    # ------------------------------------------------------------------
    # Stage 6: assemble the report
    # ------------------------------------------------------------------
    bi, bj = max(((i, j) for i in range(5) for j in range(i + 1, 5)),
                 key=lambda ij: abs(bmat[ij[0], ij[1]]))
    bcoef = float(bmat[bi, bj])
    has_int = abs(bcoef) > 1e-4

    gate_vars = []
    boundary = {"form": "linear",
                "coefficients": {v: 0.0 for v in VARS},
                "threshold": 0.0,
                "active_when": "greater_equal"}
    jump = 0.0
    disc = []
    coeffs_f = None
    tau_f = 0.0
    if gate is not None and J > 0:
        coeffs = np.asarray(gate["coeffs"], float)
        scale = float(np.max(np.abs(coeffs)))
        if scale > 0:
            coeffs = np.where(np.abs(coeffs) > 0.015 * scale, coeffs, 0.0)
            m = float(np.max(np.abs(coeffs)))
            coeffs = coeffs / m
            tau_f = float(gate["tau"]) / m
            coeffs_f = coeffs
            gate_vars = [VARS[i] for i in range(5) if abs(coeffs[i]) > 0]
            boundary = {"form": "linear",
                        "coefficients": {VARS[i]: float(coeffs[i])
                                         for i in range(5)},
                        "threshold": tau_f,
                        "active_when": "greater_equal"}
            jump = float(abs(Jf)) if abs(Jf) > 1e-9 else float(J)
            cond = " + ".join(f"({float(coeffs[i])!r})*{VARS[i]}"
                              for i in range(5) if coeffs[i] != 0)
            disc = [{"gate_variables": gate_vars,
                     "boundary": boundary,
                     "jump": jump,
                     "condition": f"{cond} >= {tau_f!r}"}]

    rel = []
    for i in range(5):
        if abs(lin[i]) > 1e-4:
            rel.append(VARS[i])
        elif has_int and i in (bi, bj):
            rel.append(VARS[i])
        elif VARS[i] in gate_vars:
            rel.append(VARS[i])
    irr = [v for v in VARS if v not in rel]

    terms = [repr(float(a0f))]
    for i in range(5):
        if abs(lin[i]) > 0.0:
            terms.append(f"({float(lin[i])!r})*x{i+1}")
    if has_int:
        terms.append(f"({bcoef!r})*x{bi+1}*x{bj+1}")
    if use_sq:
        for i in range(5):
            if abs(sq[i]) > 1e-6:
                terms.append(f"({float(sq[i])!r})*x{i+1}**2")
    expr = " + ".join(terms)
    if coeffs_f is not None:
        cond = " + ".join(f"({float(coeffs_f[i])!r})*{VARS[i]}"
                          for i in range(5) if coeffs_f[i] != 0)
        expr = expr + f" + ({float(jump)!r} if {cond} >= {tau_f!r} else 0.0)"

    pred = np.full(len(Xq), a0f)
    pred = pred + Xq @ lin
    for i in range(5):
        for j in range(i + 1, 5):
            if bmat[i, j] != 0:
                pred = pred + bmat[i, j] * Xq[:, i] * Xq[:, j]
        if use_sq and sq[i] != 0:
            pred = pred + sq[i] * Xq[:, i] ** 2
    if coeffs_f is not None:
        pred = pred + float(jump) * ((Xq @ coeffs_f) >= tau_f).astype(float)

    return {
        "relevant_variables": rel,
        "irrelevant_variables": irr,
        "interactions": ([{"variables": [VARS[bi], VARS[bj]],
                           "type": "product",
                           "coefficient": bcoef,
                           "equivalent_centered_form":
                               f"{bcoef!r}*({VARS[bi]}-0.5)*({VARS[bj]}-0.5)"}]
                         if has_int else []),
        "discontinuities": disc,
        "model_expression": expr,
        "predictions": [float(v) for v in pred],
        "notes": ("queries %d/%d. " % (qused(), budget0))
                 + "Local gradients at 7 low-discrepancy base points give the "
                   "affine gradient field (main effects + the one bilinear "
                   "term); residuals y-smooth(x) split into two exact levels and "
                   "label every reading; every dial subset is tested as a gate, "
                   "rival hypotheses are killed by readings placed where they "
                   "disagree, the survivor's surface offset is bisected to "
                   "~1e-3, and one exact least squares on "
                   "(1, x_i, x_i x_j, 1[gate]) sets all coefficients. "
                 + " | ".join(log),
    }
