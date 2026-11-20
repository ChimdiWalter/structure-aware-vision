"""Unit tests. Run from object-identity/: python -m unittest discover -s tests"""
import sys, unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from certified_identity import fixtures as FX, methods as MT, scoring as SC  # noqa: E402


class T(unittest.TestCase):
    def test_persistence_two_bumps(self):
        yy, xx = np.mgrid[0:40, 0:60].astype(float)
        f = 1.0 * np.exp(-((yy - 20) ** 2 + (xx - 15) ** 2) / 40) + 0.6 * np.exp(-((yy - 20) ** 2 + (xx - 45) ** 2) / 40)
        ms = sorted(MT.persistence_maxima(f), key=lambda m: -m["pers"])
        self.assertEqual([tuple(m["yx"]) for m in ms[:2]], [(20, 15), (20, 45)])
        saddle = f[20, 15:46].min(); self.assertAlmostEqual(ms[1]["pers"], f[20, 45] - saddle, places=9)
        self.assertTrue(ms[0]["essential"]); self.assertTrue(all(m["pers"] < 1e-6 for m in ms[2:]))

    def test_hungarian(self):
        g = np.zeros((20, 20), int); g[2:8, 2:8] = 1; g[10:18, 10:18] = 2
        self.assertEqual(SC.hungarian(g, g)["f1"], 1.0)
        p = g.copy(); p[10:18, 13:15] = 3; p[10:18, 15:18] = 4; self.assertAlmostEqual(SC.hungarian(g, p)["f1"], 2 * 1 / (2 * 1 + 3 + 1))  # pieces 24/16/24 of 64 px: no IoU >= 0.5

    def test_fixtures_build(self):
        for f in FX.FAMILIES:
            s = FX.make_scene(f, 0); self.assertEqual(s["P"].shape, (96, 96)); self.assertTrue(0 <= s["P"].min() and s["P"].max() <= 1)
        n = {f: int(len(np.unique(FX.make_scene(f, 3)["gt"])) - 1) for f in ("F1", "F2", "F3", "F5", "F5c")}
        self.assertEqual(n, {"F1": 1, "F2": 2, "F3": 1, "F5": 1, "F5c": 2})

    def test_weight_bounds_exact(self):
        s = FX.make_scene("F2", 1); rng = np.random.Generator(np.random.PCG64(1)); eta = 0.03
        lo, hi = MT.weights(s["P"], s["E"], 3, 6, eta, -1), MT.weights(s["P"], s["E"], 3, 6, eta, +1)
        for _ in range(5):
            P = np.clip(s["P"] + rng.uniform(-eta, eta, s["P"].shape), 0, 1); E = np.clip(s["E"] + rng.uniform(-eta, eta, s["E"].shape), 0, 1)
            w = MT.weights(P, E, 3, 6); self.assertTrue(np.all(lo <= w + 1e-12) and np.all(w <= hi + 1e-12))

    def test_certificate_valid(self):
        prm = {"rho": 3, "gamma": 6, "kappa": 4, "lam": 20}
        for fam, sd in (("F2", 2), ("F4", 3), ("F1", 5)):
            s = FX.make_scene(fam, sd); lab, rec = MT.select_objects(s["P"], s["E"], **prm)
            for eta in (0.01, 0.03):
                ex, cert, _ = MT.certify(s["P"], s["E"], prm, rec["sources"], rec["S"], eta); rng = np.random.Generator(np.random.PCG64(sd))
                for _ in range(4):
                    P = np.clip(s["P"] + rng.uniform(-eta, eta, s["P"].shape), 0, 1); E = np.clip(s["E"] + rng.uniform(-eta, eta, s["E"].shape), 0, 1)
                    _, r2 = MT.select_objects(P, E, sources=rec["sources"], **prm)
                    self.assertEqual(int((cert & (r2["ident"] != rec["ident"])).sum()), 0)

    def test_rotation_cc_exact(self):
        s = FX.make_scene("F4", 1); a = MT.cc(s["P"]); b = np.rot90(MT.cc(np.rot90(s["P"]).copy()), -1)
        self.assertEqual(SC.matched_disagreement(a, b), 0.0)


if __name__ == "__main__":
    unittest.main()


class Recomputed(unittest.TestCase):
    def test_rigid_certificate_holds(self):
        from certified_identity import certificates as CT
        prm = {"rho": 3, "gamma": 2, "kappa": 4, "lam": 400}; eta = 0.005
        for fam, sd in (("F1", 100), ("F3", 101)):
            s = FX.make_scene(fam, sd); f, ms, c = CT.candidates(s["P"], 6.0); ok, H, _ = CT.rigidity(f, ms, eta)
            if not ok: continue
            z, rec = MT.select_objects(s["P"], s["E"], sources=c, **prm); ex, cm = CT.certify_recomputed(s["P"], s["E"], prm, H, rec["S"], z, eta)
            rng = np.random.Generator(np.random.PCG64(sd))
            for P2, E2 in CT.adversarial_perturbations(s["P"], s["E"], z, eta, rng):
                _, _, c2 = CT.candidates(P2, 6.0); reg = [[i for i, (_, R) in enumerate(H) if R[y, x]] for y, x in c2]
                self.assertEqual(sorted(g[0] for g in reg if len(g) == 1), list(range(len(H))))
                src = [None] * len(H)
                for yx, g in zip(c2, reg): src[g[0]] = yx
                self.assertEqual(int((cm & (MT.select_objects(P2, E2, sources=src, **prm)[1]["ident"] != rec["ident"])).sum()), 0)
