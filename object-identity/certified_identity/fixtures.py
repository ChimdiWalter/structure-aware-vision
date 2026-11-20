"""Synthetic fixtures. Deterministic from (family, seed).

Each scene: P (foreground evidence), E (boundary evidence), gt (instance labels, 0 = background). Nothing is hand-edited.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi

N = 96
FAMILIES = ("F1", "F2", "F2w", "F3", "F4", "F5", "F5c")


def _yx(h, w):
    return np.mgrid[0:h, 0:w].astype(np.float64)


def ellipse(h, w, cy, cx, ry, rx, th):
    yy, xx = _yx(h, w); c, s = np.cos(th), np.sin(th)
    u = (xx - cx) * c + (yy - cy) * s; v = -(xx - cx) * s + (yy - cy) * c
    return (u / rx) ** 2 + (v / ry) ** 2 <= 1.0


def band(h, w, py, px, ang, half=1.0):
    yy, xx = _yx(h, w)
    return np.abs((xx - px) * np.sin(ang) - (yy - py) * np.cos(ang)) <= half


def snoise(rng, h, w, sigma=4.0):
    n = ndi.gaussian_filter(rng.normal(size=(h, w)), sigma); return n / (n.std() + 1e-12)


def boundary(lab, other=None):
    """Pixels with label > 0 that have a 4-neighbour with a different label (only neighbours == other, if given)."""
    p = np.pad(lab, 1, mode="edge"); c = p[1:-1, 1:-1]; out = np.zeros(lab.shape, bool)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        nb = p[1 + dy:1 + dy + lab.shape[0], 1 + dx:1 + dx + lab.shape[1]]
        m = (c > 0) & (nb != c)
        out |= (m & (nb == other)) if other is not None else m
    return out


class Canvas:
    def __init__(self, rng, h, w):
        self.rng, self.h, self.w = rng, h, w
        self.P = np.clip(0.08 + 0.03 * snoise(rng, h, w), 0.01, 0.3)
        self.E = 0.04 * np.abs(snoise(rng, h, w, 2.0))
        self.gt = np.zeros((h, w), np.int32)

    def paint_P(self, mask, level, amp=0.03, soft=1.0):
        a = ndi.gaussian_filter(mask.astype(float), soft) if soft > 0 else mask.astype(float)
        self.P = self.P * (1 - a) + (level + amp * snoise(self.rng, self.h, self.w)) * a

    def paint_E(self, mask, strength):
        if mask.any():
            m = ndi.gaussian_filter(mask.astype(float), 0.7); self.E = np.maximum(self.E, strength * m / m.max())

    def add_object(self, mask, label=None):
        self.paint_P(mask, 0.88); self.paint_E(boundary(mask.astype(np.int32)), self.rng.uniform(0.6, 0.9))
        if label is not None: self.gt[mask] = label

    def scene(self, **meta):
        return {"P": np.clip(self.P, 0.0, 1.0), "E": np.clip(self.E, 0.0, 1.0), "gt": self.gt.copy(), **meta}


def waist_pair(rng, h, w, cy, cx):
    r1, r2 = rng.uniform(9, 13, 2); phi = rng.uniform(0, np.pi); D = (r1 + r2) * rng.uniform(0.72, 0.85)
    c1 = (cy - 0.5 * D * np.sin(phi), cx - 0.5 * D * np.cos(phi)); c2 = (cy + 0.5 * D * np.sin(phi), cx + 0.5 * D * np.cos(phi))
    A = ellipse(h, w, c1[0], c1[1], r1 * rng.uniform(0.85, 1.0), r1, rng.uniform(0, np.pi))
    B = ellipse(h, w, c2[0], c2[1], r2 * rng.uniform(0.85, 1.0), r2, rng.uniform(0, np.pi))
    yy, xx = _yx(h, w); U = A | B
    d1 = np.hypot(yy - c1[0], xx - c1[1]) / r1; d2 = np.hypot(yy - c2[0], xx - c2[1]) / r2
    lab = np.zeros((h, w), np.int32); lab[U & (d1 <= d2)] = 1; lab[U & (d1 > d2)] = 2
    return U, lab


def _place(rng, h, w, occupied, make, gap=4, tries=300):
    for _ in range(tries):
        cy, cx = rng.uniform(16, h - 16), rng.uniform(16, w - 16); obj = make(cy, cx)
        m = obj[0] if isinstance(obj, tuple) else obj
        if m.sum() < 40 or m[:2].any() or m[-2:].any() or m[:, :2].any() or m[:, -2:].any(): continue
        if not (ndi.binary_dilation(m, iterations=gap) & occupied).any(): return obj
    raise RuntimeError("placement failed")


def make_scene(family, seed):
    rng = np.random.Generator(np.random.PCG64([20261111, seed, FAMILIES.index(family) if family in FAMILIES else 99]))
    h = w = N
    for _ in range(50):
        cv = Canvas(rng, h, w)
        if family in ("F1", "F2", "F2w"):
            U, lab = waist_pair(rng, h, w, rng.uniform(40, 56), rng.uniform(40, 56))
            cv.add_object(U)
            if family == "F1": cv.gt[U] = 1
            else:
                cv.gt[:] = lab; contact = boundary(lab, 2) | boundary(lab, 1)
                cv.paint_E(contact, rng.uniform(0.6, 0.9) if family == "F2" else rng.uniform(0.05, 0.2))
        elif family == "F3":
            th = rng.uniform(0, np.pi); cy, cx = rng.uniform(40, 56, 2)
            el = ellipse(h, w, cy, cx, rng.uniform(9, 12), rng.uniform(14, 19), th); cv.add_object(el, 1)
            vein = band(h, w, cy + rng.uniform(-3, 3), cx + rng.uniform(-3, 3), rng.uniform(0, np.pi)) & ndi.binary_dilation(el, iterations=2)
            cv.paint_E(vein, rng.uniform(0.6, 0.9))
        elif family == "F4":
            occ = np.zeros((h, w), bool); k = 0
            for _i in range(int(rng.integers(1, 3))):
                r = rng.uniform(8, 13); m = _place(rng, h, w, occ, lambda y, x: ellipse(h, w, y, x, r * rng.uniform(0.7, 1.0), r, rng.uniform(0, np.pi)))
                k += 1; cv.add_object(m, k); occ |= m
            for _i in range(2):
                r = rng.uniform(6, 11); m = _place(rng, h, w, occ, lambda y, x: ellipse(h, w, y, x, r * rng.uniform(0.7, 1.0), r, rng.uniform(0, np.pi)))
                cv.paint_P(m, rng.uniform(0.55, 0.68), amp=0.015); cv.paint_E(boundary(m.astype(np.int32)), rng.uniform(0.1, 0.25)); occ |= m
        elif family in ("F5", "F5c"):
            th = rng.uniform(0, np.pi); cy, cx = rng.uniform(40, 56, 2)
            el = ellipse(h, w, cy, cx, rng.uniform(7, 10), rng.uniform(18, 25), th); off = rng.uniform(-4, 4)
            occl = band(h, w, cy + off * np.sin(th), cx + off * np.cos(th), th + np.pi / 2 + rng.uniform(-0.35, 0.35)) & el
            frag = el & ~occl; lab, n = ndi.label(frag, structure=np.ones((3, 3), bool))
            if n != 2 or min(int((lab == 1).sum()), int((lab == 2).sum())) < 60: continue
            cv.add_object(el); cv.paint_P(occl, 0.15, amp=0.0, soft=0.5)
            cv.paint_E(boundary(lab, 0) & ndi.binary_dilation(occl, iterations=1), rng.uniform(0.4, 0.6))
            cv.gt[:] = (frag.astype(np.int32) if family == "F5" else lab.astype(np.int32))
        else:
            raise ValueError(family)
        return cv.scene(family=family, seed=seed)
    raise RuntimeError(f"could not build {family} seed {seed}")


def make_tiling_scene(seed, size=160):
    """F6: 8 mixed objects (single, waist-one, touching pair x2, internal-line, single, false blob x2)."""
    rng = np.random.Generator(np.random.PCG64([20261111, seed, 6])); h = w = size; cv = Canvas(rng, h, w); occ = np.zeros((h, w), bool); k = 0
    for kind in ("one", "waist1", "pair", "pair", "line", "false", "one", "false"):
        if kind in ("waist1", "pair"):
            U, lab = _place(rng, h, w, occ, lambda y, x: waist_pair(rng, h, w, y, x)); cv.add_object(U)
            if kind == "waist1": k += 1; cv.gt[U] = k
            else:
                cv.gt[lab == 1] = k + 1; cv.gt[lab == 2] = k + 2; k += 2
                cv.paint_E(boundary(lab, 2) | boundary(lab, 1), rng.uniform(0.6, 0.9))
            occ |= U
        elif kind == "line":
            th = rng.uniform(0, np.pi); ry, rx = rng.uniform(9, 12), rng.uniform(14, 19)
            el = _place(rng, h, w, occ, lambda y, x: ellipse(h, w, y, x, ry, rx, th)); k += 1; cv.add_object(el, k); occ |= el
            ys, xs = np.nonzero(el); vein = band(h, w, ys.mean(), xs.mean(), rng.uniform(0, np.pi)) & ndi.binary_dilation(el, iterations=2)
            cv.paint_E(vein, rng.uniform(0.6, 0.9))
        elif kind == "one":
            r = rng.uniform(8, 13); m = _place(rng, h, w, occ, lambda y, x: ellipse(h, w, y, x, r * rng.uniform(0.7, 1.0), r, rng.uniform(0, np.pi)))
            k += 1; cv.add_object(m, k); occ |= m
        else:
            r = rng.uniform(6, 11); m = _place(rng, h, w, occ, lambda y, x: ellipse(h, w, y, x, r * rng.uniform(0.7, 1.0), r, rng.uniform(0, np.pi)))
            cv.paint_P(m, rng.uniform(0.55, 0.68), amp=0.015); cv.paint_E(boundary(m.astype(np.int32)), rng.uniform(0.1, 0.25)); occ |= m
    return cv.scene(family="F6", seed=seed)


def crops(scene, size=96, offsets=(0, 32, 64)):
    out = []
    for oy in offsets:
        for ox in offsets:
            sl = (slice(oy, oy + size), slice(ox, ox + size))
            out.append({"P": scene["P"][sl].copy(), "E": scene["E"][sl].copy(), "gt": scene["gt"][sl].copy(), "offset": (oy, ox)})
    return out
