"""Object agreement, hypothesis-creation, consistency and identity-change measures ."""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

OVL = 20


def _inter(a, b):
    m = (a > 0) & (b > 0); B = int(b.max()) + 1
    k, c = np.unique(a[m].astype(np.int64) * B + b[m], return_counts=True)
    return {(int(x // B), int(x % B)): int(n) for x, n in zip(k, c)}


def hungarian(gt, pred, thr=0.5):
    g = [int(v) for v in np.unique(gt) if v > 0]; p = [int(v) for v in np.unique(pred) if v > 0]
    if not g and not p: return {"f1": 1.0, "tp": 0, "fp": 0, "fn": 0, "n_gt": 0, "n_pred": 0}
    ga, pa = np.bincount(gt.ravel()), np.bincount(pred.ravel()); I = _inter(gt, pred); tp = 0
    if g and p:
        M = np.zeros((len(g), len(p)))
        for (i, j), n in I.items(): M[g.index(i), p.index(j)] = n / (ga[i] + pa[j] - n)
        r, c = linear_sum_assignment(-M); tp = int((M[r, c] >= thr).sum())
    fp, fn = len(p) - tp, len(g) - tp
    return {"f1": 2 * tp / (2 * tp + fp + fn), "tp": tp, "fp": fp, "fn": fn, "n_gt": len(g), "n_pred": len(p)}


def categories(gt, pred, ovl=OVL):
    I = _inter(gt, pred); g2p, p2g = {}, {}
    for (i, j), n in I.items():
        if n >= ovl: g2p.setdefault(i, set()).add(j); p2g.setdefault(j, set()).add(i)
    out = {"correct": 0, "fragmented": 0, "fused": 0, "complex": 0, "missed": 0}
    for g in (int(v) for v in np.unique(gt) if v > 0):
        ps = g2p.get(g, set())
        if not ps: out["missed"] += 1
        elif len(ps) == 1 and len(p2g[next(iter(ps))]) == 1: out["correct"] += 1
        elif len(ps) > 1 and all(len(p2g[q]) == 1 for q in ps): out["fragmented"] += 1
        elif len(ps) == 1: out["fused"] += 1
        else: out["complex"] += 1
    out["false_objects"] = sum(1 for j in (int(v) for v in np.unique(pred) if v > 0) if j not in p2g)
    return out


def versus_cc(P, lab):
    """How selection output departs from connected components of P >= 0.5."""
    cc = ndi.label(P >= 0.5, structure=np.ones((3, 3), bool))[0]; I = _inter(cc, lab); splits = rejects = 0
    for c in (int(v) for v in np.unique(cc) if v > 0):
        objs = [j for (i, j), n in I.items() if i == c and n >= OVL]
        if len(objs) >= 2: splits += 1
        if (lab[cc == c] == 0).mean() >= 0.8: rejects += 1
    merges = sum(1 for j in (int(v) for v in np.unique(lab) if v > 0) if len([i for (i, jj), n in I.items() if jj == j and n >= OVL]) >= 2)
    return {"splits": splits, "merges": merges, "rejections": rejects, "boundary_move_frac": float(((lab > 0) != (P >= 0.5)).mean())}


def matched_disagreement(a, b):
    """Share of pixels foreground in either labelling whose identity differs after optimal (IoU) matching of b's labels onto a's."""
    U = (a > 0) | (b > 0)
    if not U.any(): return 0.0
    ia = [int(v) for v in np.unique(a) if v > 0]; ib = [int(v) for v in np.unique(b) if v > 0]; mp = {}
    if ia and ib:
        I = _inter(a, b); M = np.zeros((len(ia), len(ib)))
        for (i, j), n in I.items(): M[ia.index(i), ib.index(j)] = n
        r, c = linear_sum_assignment(-M)
        for i, j in zip(r, c):
            if M[i, j] > 0: mp[ib[j]] = ia[i]
    bb = np.zeros_like(b); nxt = (max(ia) if ia else 0) + 1
    for j in ib:
        if j in mp: bb[b == j] = mp[j]
        else: bb[b == j] = nxt; nxt += 1
    return float((a[U] != bb[U]).mean())


def overlap_ari(la, oa, lb, ob, size):
    from sklearn.metrics import adjusted_rand_score
    y0, x0 = max(oa[0], ob[0]), max(oa[1], ob[1]); y1, x1 = min(oa[0], ob[0]) + size, min(oa[1], ob[1]) + size
    if y1 <= y0 or x1 <= x0: return None
    A = la[y0 - oa[0]:y1 - oa[0], x0 - oa[1]:x1 - oa[1]]; B = lb[y0 - ob[0]:y1 - ob[0], x0 - ob[1]:x1 - ob[1]]; U = (A > 0) | (B > 0)
    return float(adjusted_rand_score(A[U], B[U])) if U.sum() > 0 else None


def bootstrap(vals, seed=20261111, B=2000):
    v = np.asarray(vals, float)
    if v.size == 0: return {"n": 0}
    rng = np.random.Generator(np.random.PCG64(seed)); bs = v[rng.integers(0, v.size, (B, v.size))].mean(1)
    return {"n": int(v.size), "mean": round(float(v.mean()), 4), "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
