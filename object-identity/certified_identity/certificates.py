"""Certificates for object existence and membership (see docs/theory.md).

- paired_existence / pixel_certificate: candidates fixed; interval geodesic costs; the existence decision is certified by bounding
  the DIFFERENCE of objectives within one perturbed realization (exact cancellation where a shared label provably wins).
- rigidity / certify_recomputed: candidates RECOMPUTED from the perturbed field; certified only in rigid configurations
  (persistence gap, value margin, disjoint localization regions), using uncertain-source cost intervals.
- threshold_band: the simple comparator, a band certificate for filtered connected components.
- persistence-based proposals (psi_candidates) and an evidence merge rule (merge_rule) used in the demos.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from . import methods as MT

S8 = np.ones((3, 3), bool)


def _paired(LO, HI, S, K, lam):
    LS = [0] + [i + 1 for i in S]
    for mask in range(1 << K):
        T = tuple(i for i in range(K) if (mask >> i) & 1)
        if T == tuple(S): continue
        LT = [0] + [j + 1 for j in T]; U = sorted(set(LS) | set(LT)); ub = HI[LS].min(0) - LO[LT].min(0)
        for l in set(LS) & set(LT):
            oth = [u for u in U if u != l]
            if oth: ub = np.where(HI[l] < LO[oth].min(0), 0.0, ub)
        if ub.sum() + lam * (len(S) - len(T)) >= 0: return False
    return True


def _pixels(LO, HI, S, z):
    rows = np.array([0] + [i + 1 for i in S]); zz = z.ravel(); cols = np.arange(zz.size); lo, hi = LO[rows], HI[rows]
    lo2 = lo.copy(); lo2[zz, cols] = np.inf
    return hi[zz, cols] < (lo2.min(0) if lo.shape[0] > 1 else np.inf)


def certify_fixed(P, E, prm, src, S, eta):
    """Fixed candidates. Returns (existence certified, certified-pixel mask)."""
    Clo, c0lo = MT.costs(P, E, src, prm["rho"], prm["gamma"], prm["kappa"], eta, -1); Chi, c0hi = MT.costs(P, E, src, prm["rho"], prm["gamma"], prm["kappa"], eta, +1)
    LO = np.vstack([c0lo[None, :], Clo]); HI = np.vstack([c0hi[None, :], Chi]); ex = _paired(LO, HI, S, Clo.shape[0], prm["lam"])
    Cn, c0n = MT.costs(P, E, src, prm["rho"], prm["gamma"], prm["kappa"]); z, _, _ = MT.assign(Cn, c0n, S)
    return ex, (_pixels(LO, HI, S, z) & ex).reshape(P.shape)


def candidates(P, sigma, tau=0.06, fmin=0.35, K=10):
    f = ndi.gaussian_filter(np.asarray(P, np.float64), sigma); ms = MT.persistence_maxima(f)
    c = sorted((m for m in ms if m["pers"] >= tau and f[m["yx"]] >= fmin), key=lambda m: (-m["pers"], m["yx"]))
    return f, ms, [tuple(int(v) for v in m["yx"]) for m in c[:K]]


def rigidity(f, ms, eta, tau=0.06, fmin=0.35, K=10):
    """Proposition B2 conditions. Returns (rigid, [(maximum, localization region)] for high-value robust maxima, info)."""
    band = [m for m in ms if not m["essential"] and tau - 2 * eta <= m["pers"] <= tau + 2 * eta]
    robust = [m for m in ms if m["essential"] or m["pers"] > tau + 2 * eta]; regions = []
    for m in robust:
        if m["essential"]: B = np.ones(f.shape, bool)
        else: lab = ndi.label(f > m["birth"] - m["pers"], structure=S8)[0]; B = lab == lab[m["yx"]]
        regions.append(B & (f >= f[m["yx"]] - 2 * eta))
    r2 = all(f[m["yx"]] >= fmin + eta or f[m["yx"]] < fmin - eta for m in robust)
    disjoint = all(not (regions[i] & regions[j]).any() for i in range(len(regions)) for j in range(i + 1, len(regions)))
    H = sorted(((m, R) for m, R in zip(robust, regions) if f[m["yx"]] >= fmin + eta), key=lambda t: (-t[0]["pers"], t[0]["yx"]))
    return (not band) and r2 and disjoint and len(H) <= K and tau > 4 * eta, H, {"band": len(band), "robust": len(robust)}


def certify_recomputed(P, E, prm, H, S, z, eta):
    """Theorem B4: uncertain-source intervals [min over region of d_lo, d_hi(m, .) + r_m], then paired existence + pixel certificates."""
    u, v, _ = MT.edges(*P.shape); n = P.size; W = P.shape[1]
    G = lambda w: csr_matrix((np.r_[w, w], (np.r_[u, v], np.r_[v, u])), shape=(n, n))
    Glo = G(MT.weights(P, E, prm["rho"], prm["gamma"], eta, -1)); Ghi = G(MT.weights(P, E, prm["rho"], prm["gamma"], eta, +1)); Clo, Chi = [], []
    for m, R in H:
        idx = np.flatnonzero(R.ravel()); Clo.append(dijkstra(Glo, directed=True, indices=idx, min_only=True))
        dm = dijkstra(Ghi, directed=True, indices=[m["yx"][0] * W + m["yx"][1]])[0]; Chi.append(dm + dm[idx].max())
    Pv = P.ravel(); LO = np.vstack([prm["kappa"] * np.maximum(Pv - eta, 0)[None, :]] + Clo); HI = np.vstack([prm["kappa"] * np.minimum(Pv + eta, 1)[None, :]] + Chi)
    ex = _paired(LO, HI, S, len(H), prm["lam"])
    return ex, (_pixels(LO, HI, S, z) & ex).reshape(P.shape)


def threshold_band(P, eta, p_min=0.7):
    """Comparator: certified components of {P >= 0.5} filtered by mean P >= p_min. Returns (labels, {object id: (core, region)})."""
    lab = MT.ccf(P, p_min); lq, nq = ndi.label(P >= 0.5 - eta, structure=S8); lc, _ = ndi.label(P >= 0.5 + eta, structure=S8); cert = {}
    for q in range(1, nq + 1):
        Q = lq == q; cs = [int(x) for x in np.unique(lc[Q]) if x > 0]
        if len(cs) != 1: continue
        C = lc == cs[0]
        if (Q & ~ndi.binary_dilation(C, structure=S8)).any(): continue
        band = Q & ~C; nC = int(C.sum()); lo_b = np.sort(np.maximum(P[band] - eta, 0.5))
        sl = float(np.maximum(P[C] - eta, 0).sum())
        if min((sl + float(lo_b[:k].sum())) / (nC + k) for k in range(len(lo_b) + 1)) >= p_min:
            ids = [int(x) for x in np.unique(lab[C])]
            if len(ids) == 1 and ids[0] > 0: cert[ids[0]] = (C, Q)
    return lab, cert


def adversarial_perturbations(P, E, z, eta, rng):
    """Uniform, vertex (+/- eta), global shifts, object-targeted and boundary-targeted perturbations of (P, E), clipped to [0, 1]."""
    out = [(P + rng.uniform(-eta, eta, P.shape), E + rng.uniform(-eta, eta, E.shape)) for _ in range(2)]
    out += [(P + eta * rng.choice((-1.0, 1.0), P.shape), E + eta * rng.choice((-1.0, 1.0), E.shape)) for _ in range(3)]
    out += [(P - eta, E + eta), (P + eta, E - eta)]
    for a in [x for x in np.unique(z) if x > 0][:4]:
        sg = np.where(ndi.binary_dilation(z == a, iterations=2), 1.0, -1.0); out += [(P - eta * sg, E + eta * sg), (P + eta * sg, E - eta * sg)]
    return [(np.clip(p, 0, 1), np.clip(e, 0, 1)) for p, e in out]


def psi_candidates(P, tau=0.1, K=10):
    """Over-complete shape proposals: persistence maxima of P x (per-component normalized distance transform)."""
    sup = P >= 0.5; psi = P * MT.d_field(sup); ms = [m for m in MT.persistence_maxima(psi, sup) if m["essential"] or m["pers"] >= tau]
    ms.sort(key=lambda m: (-m["pers"], m["yx"])); return [tuple(int(x) for x in m["yx"]) for m in ms[:K]]


def merge_rule(base, P, E, e_merge=0.3, p_min=0.7):
    """Non-geometric comparator: merge adjacent regions whose shared-boundary mean evidence E < e_merge; drop regions with mean P < p_min."""
    lab = base; keys, vals = [], []
    for a, b, ea, eb in ((lab[:, :-1], lab[:, 1:], E[:, :-1], E[:, 1:]), (lab[:-1, :], lab[1:, :], E[:-1, :], E[1:, :])):
        m = (a > 0) & (b > 0) & (a != b); keys.append(np.minimum(a[m], b[m]).astype(np.int64) * 100000 + np.maximum(a[m], b[m])); vals.append(0.5 * (ea[m] + eb[m]))
    keys, vals = np.concatenate(keys), np.concatenate(vals); par = list(range(int(lab.max()) + 1))

    def fnd(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    if keys.size:
        uk, inv = np.unique(keys, return_inverse=True); mean = np.bincount(inv, vals) / np.bincount(inv)
        for k, mu in zip(uk, mean):
            if mu < e_merge:
                a, b = fnd(int(k // 100000)), fnd(int(k % 100000))
                if a != b: par[max(a, b)] = min(a, b)
    out = np.zeros(lab.shape, np.int32)
    for i in range(1, int(lab.max()) + 1): out[lab == i] = fnd(i)
    for x in np.unique(out[out > 0]):
        if P[out == x].mean() < p_min: out[out == x] = 0
    return MT._relabel(out)
