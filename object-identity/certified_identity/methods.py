"""Partition mechanisms and certificates.

CC, CCF, WS_psi, WS_blur, RM: cheap baselines. selection: persistence candidates on f = G_4*P, an evidence metric on the 8-neighbour graph,
geodesic assignment costs (scipy csgraph Dijkstra), explicit background cost, existence by exact facility location over candidate subsets.
"""
from __future__ import annotations
import math
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

S8 = np.ones((3, 3), bool); SQ2 = math.sqrt(2.0); EPS0 = 0.05; K_MAX = 10; CAND_TAU = 0.02; CAND_FMIN = 0.35


def blur(P): return ndi.gaussian_filter(np.asarray(P, np.float64), 4.0)


def persistence_maxima(f, mask=None):
    """Superlevel-set H0 persistence of f (8-connectivity) on mask. One record per maximum:
    pers = birth - death (elder rule; ties by smaller pixel index); essential maxima get pers = birth - component minimum."""
    h, w = f.shape; fv = np.asarray(f, np.float64).ravel(); n = h * w
    act = np.ones(n, bool) if mask is None else np.asarray(mask, bool).ravel()
    idx = np.flatnonzero(act); order = idx[np.lexsort((idx, -fv[idx]))]
    parent = np.full(n, -1, np.int64); bidx = np.full(n, -1, np.int64); out = []

    def find(a):
        r = a
        while parent[r] != r: r = parent[r]
        while parent[a] != r: parent[a], a = r, parent[a]
        return r
    for p in order:
        y, x = divmod(int(p), w); roots = set()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dy or dx) and 0 <= y + dy < h and 0 <= x + dx < w:
                    q = (y + dy) * w + x + dx
                    if parent[q] >= 0: roots.add(find(q))
        if not roots:
            parent[p] = p; bidx[p] = p; continue
        rl = sorted(roots, key=lambda r: (-fv[bidx[r]], bidx[r])); elder = rl[0]
        for r in rl[1:]:
            out.append({"pers": float(fv[bidx[r]] - fv[p]), "birth": float(fv[bidx[r]]), "yx": divmod(int(bidx[r]), w), "essential": False})
            parent[r] = elder
        parent[p] = elder
    if idx.size:
        comp = np.array([find(int(p)) for p in idx]); mins = {}
        for r, v in zip(comp, fv[idx]): mins[r] = min(mins.get(r, np.inf), v)
        for r, m in mins.items():
            out.append({"pers": float(fv[bidx[r]] - m), "birth": float(fv[bidx[r]]), "yx": divmod(int(bidx[r]), w), "essential": True})
    return out


def _marker_ws(cost, seeds, mask):
    from skimage.segmentation import watershed
    mk = np.zeros(mask.shape, np.int32)
    for i, (y, x) in enumerate(sorted(set(seeds)), start=1):
        if mask[y, x]: mk[y, x] = i
    return watershed(cost, markers=mk, mask=mask, connectivity=2).astype(np.int32)


def _relabel(lab):
    u = np.unique(lab); u = u[u > 0]; out = np.zeros(lab.shape, np.int32)
    for i, v in enumerate(u, start=1): out[lab == v] = i
    return out


def cc(P, **_):
    return ndi.label(P >= 0.5, structure=S8)[0].astype(np.int32)


def ccf(P, p_min, **_):
    lab = cc(P)
    for i in range(1, int(lab.max()) + 1):
        if P[lab == i].mean() < p_min: lab[lab == i] = 0
    return _relabel(lab)


def d_field(sup):
    lab, n = ndi.label(sup, structure=S8); d = np.zeros(sup.shape)
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        own = lab[sl] == i; e = ndi.distance_transform_edt(np.pad(own, 1))[1:-1, 1:-1]; d[sl][own] = e[own] / max(e.max(), 1.0)
    return d


def ws_psi(P, tau, **_):
    sup = P >= 0.5; psi = P * d_field(sup)
    seeds = [m["yx"] for m in persistence_maxima(psi, sup) if m["essential"] or m["pers"] >= tau]
    return _marker_ws(-psi, seeds, sup)


def ws_blur(P, tau, **_):
    sup = P >= 0.5; f = blur(P)
    seeds = [m["yx"] for m in persistence_maxima(f, sup) if m["essential"] or m["pers"] >= tau]
    return _marker_ws(-f, seeds, sup)


def rm(P, E, e_merge, p_min, **_):
    lab = ws_blur(P, CAND_TAU); keys, vals = [], []
    for a, b, ea, eb in ((lab[:, :-1], lab[:, 1:], E[:, :-1], E[:, 1:]), (lab[:-1, :], lab[1:, :], E[:-1, :], E[1:, :])):
        m = (a > 0) & (b > 0) & (a != b)
        keys.append(np.minimum(a[m], b[m]).astype(np.int64) * 100000 + np.maximum(a[m], b[m])); vals.append(0.5 * (ea[m] + eb[m]))
    keys, vals = np.concatenate(keys), np.concatenate(vals); par = list(range(int(lab.max()) + 1))

    def fnd(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    if keys.size:
        uk, inv = np.unique(keys, return_inverse=True); mean = np.bincount(inv, vals) / np.bincount(inv)
        for k, mu in sorted(zip(uk, mean), key=lambda t: (t[1], t[0])):
            if mu < e_merge:
                a, b = fnd(int(k // 100000)), fnd(int(k % 100000))
                if a != b: par[max(a, b)] = min(a, b)
    out = np.zeros(lab.shape, np.int32)
    for i in range(1, int(lab.max()) + 1): out[lab == i] = fnd(i)
    for v in np.unique(out[out > 0]):
        if P[out == v].mean() < p_min: out[out == v] = 0
    return _relabel(out)


# ---------------- selection ----------------
_EDGES = {}


def edges(h, w):
    if (h, w) not in _EDGES:
        ii = np.arange(h * w).reshape(h, w); us, vs, ls = [], [], []
        for dy, dx, l in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, SQ2), (1, -1, SQ2)):
            x0, x1 = (0, w - dx) if dx >= 0 else (-dx, w)
            a = ii[0:h - dy, x0:x1].ravel(); b = ii[dy:h, x0 + dx:x1 + dx].ravel()
            us.append(a); vs.append(b); ls.append(np.full(a.size, l))
        _EDGES[(h, w)] = (np.concatenate(us), np.concatenate(vs), np.concatenate(ls))
    return _EDGES[(h, w)]


def weights(P, E, rho, gamma, eta=0.0, side=0):
    """w = l [eps0 + rho a^2 + gamma b^2], a = 1 - mean P, b = mean E on each edge. side -1/+1: exact lower/upper bound
    over all fields with |dP|, |dE| <= eta (a, b stay in [0, 1] and squaring is monotone there)."""
    u, v, l = edges(*P.shape); Pv, Ev = np.asarray(P).ravel(), np.asarray(E).ravel()
    a = 1.0 - 0.5 * (Pv[u] + Pv[v]); b = 0.5 * (Ev[u] + Ev[v])
    if side < 0: a, b = np.maximum(a - eta, 0.0), np.maximum(b - eta, 0.0)
    elif side > 0: a, b = np.minimum(a + eta, 1.0), np.minimum(b + eta, 1.0)
    return l * (EPS0 + rho * a * a + gamma * b * b)


def geodesic(shape, wts, sources):
    h, w = shape; n = h * w; u, v, _ = edges(h, w)
    if not sources: return np.zeros((0, n))
    G = csr_matrix((np.concatenate([wts, wts]), (np.concatenate([u, v]), np.concatenate([v, u]))), shape=(n, n))
    return dijkstra(G, directed=True, indices=[y * w + x for y, x in sources])


def candidates(P, tau=CAND_TAU, kmax=K_MAX, fmin=CAND_FMIN):
    f = blur(P); ms = [m for m in persistence_maxima(f) if m["pers"] >= tau and f[m["yx"]] >= fmin]
    ms.sort(key=lambda m: (-m["pers"], m["yx"]))
    return [tuple(int(v) for v in m["yx"]) for m in ms[:kmax]]


def costs(P, E, sources, rho, gamma, kappa, eta=0.0, side=0):
    C = geodesic(P.shape, weights(P, E, rho, gamma, eta, side), sources); Pv = np.asarray(P).ravel()
    c0 = kappa * (np.maximum(Pv - eta, 0.0) if side < 0 else np.minimum(Pv + eta, 1.0) if side > 0 else Pv)
    return C, c0


def subset_sums(C, c0):
    """sum_x min_{i in S+{0}} c_i(x) for every subset S of the K candidates (exact enumeration, K <= K_MAX)."""
    out = []
    for m in range(1 << C.shape[0]):
        S = tuple(i for i in range(C.shape[0]) if (m >> i) & 1)
        out.append((S, float((c0 if not S else np.minimum(c0, C[list(S)].min(axis=0))).sum())))
    return out


def pick(sums, lam):
    """Facility location: F(S) = subset sum + lam |S|. Returns (S*, table). Ties: smaller F, then smaller |S|, then lexicographic S."""
    table = {S: base + lam * len(S) for S, base in sums}
    return min((F, len(S), S) for S, F in table.items())[2], table


def select(C, c0, lam):
    return pick(subset_sums(C, c0), lam)


def assign(C, c0, S):
    st = np.vstack([c0[None, :]] + ([C[list(S)]] if S else []))
    z = np.argmin(st, axis=0); srt = np.sort(st, axis=0)
    margin = srt[1] - srt[0] if st.shape[0] > 1 else np.full(c0.shape, np.inf)
    ident = np.where(z > 0, np.array((-1,) + tuple(S))[z], -1)       # candidate index, -1 = background
    return z, margin, ident


def select_objects(P, E, rho, gamma, kappa, lam, sources=None, **_):
    if sources is None: sources = candidates(P)
    C, c0 = costs(P, E, sources, rho, gamma, kappa); S, table = select(C, c0, lam); z, margin, ident = assign(C, c0, S)
    rec = {"sources": sources, "S": S, "margin": margin.reshape(P.shape), "ident": ident.reshape(P.shape), "table": table}
    return z.reshape(P.shape).astype(np.int32), rec


def certify(P, E, prm, sources, S, eta):
    """Interval certificates for |dP|, |dE| <= eta with candidates fixed. Returns (existence certified, pixel-certified mask)."""
    Clo, c0lo = costs(P, E, sources, prm["rho"], prm["gamma"], prm["kappa"], eta, -1)
    Chi, c0hi = costs(P, E, sources, prm["rho"], prm["gamma"], prm["kappa"], eta, +1)
    _, tlo = select(Clo, c0lo, prm["lam"]); _, thi = select(Chi, c0hi, prm["lam"])
    others = [F for T, F in tlo.items() if T != S]; exist = thi[S] < (min(others) if others else np.inf)
    Cn, c0n = costs(P, E, sources, prm["rho"], prm["gamma"], prm["kappa"]); z, _, _ = assign(Cn, c0n, S)
    lo = np.vstack([c0lo[None, :]] + ([Clo[list(S)]] if S else [])); hi = np.vstack([c0hi[None, :]] + ([Chi[list(S)]] if S else []))
    cols = np.arange(z.size); hw = hi[z, cols]; lo2 = lo.copy(); lo2[z, cols] = np.inf
    pix = hw < lo2.min(axis=0) if lo.shape[0] > 1 else np.ones(z.size, bool)
    return bool(exist), (pix & exist).reshape(P.shape), {"F_hi_star": thi[S], "F_lo_best_other": min(others) if others else None}


METHODS = {"CC": cc, "CCF": ccf, "WS_psi": ws_psi, "WS_blur": ws_blur, "RM": rm}
GRIDS = {"CC": [{}], "CCF": [{"p_min": v} for v in (0.6, 0.7, 0.75, 0.8)], "WS_psi": [{"tau": v} for v in (0.05, 0.1, 0.2, 0.3)],
         "WS_blur": [{"tau": v} for v in (0.02, 0.05, 0.1, 0.2)],
         "RM": [{"e_merge": e, "p_min": p} for e in (0.3, 0.45, 0.6) for p in (0.5, 0.7, 0.75)],
         "selection": [{"rho": r, "gamma": g, "kappa": k, "lam": l} for r in (1, 3) for g in (2, 6) for k in (2, 4) for l in (5, 20, 60)]}


def run_method(name, scene, prm, sources=None):
    if name == "selection": return select_objects(scene["P"], scene["E"], sources=sources, **prm)
    return METHODS[name](P=scene["P"], E=scene["E"], **prm), {}
