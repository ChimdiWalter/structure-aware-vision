"""Synthetic demo (a few minutes on one CPU core): object formation and certificates.

  python demo.py            # 5 scenes per family
  python demo.py --scenes 30

Part 1 compares connected components, a persistence-seeded watershed, an evidence merge rule, and facility-location selection over
over-complete topological proposals with geodesic assignment. Part 2 checks the certificates against adversarial perturbations:
fixed candidates (paired existence certificate) and recomputed candidates (rigid configurations), next to a threshold-band
certificate. A certificate violation is a certified pixel whose object identity changes; it must never happen.
"""
import argparse
import numpy as np
from certified_identity import fixtures as FX, methods as MT, scoring as SC, certificates as CT

PRM = {"rho": 3, "gamma": 2, "kappa": 4, "lam": 400}
NAMES = {"F1": "one irregular object", "F2": "two touching, contact visible", "F2w": "two touching, contact faint",
         "F3": "one object crossed by a line", "F4": "objects + faint false blobs", "F5": "one object in two fragments",
         "F5c": "same evidence, two objects"}


def part1(n):
    print("\nPart 1: object F1 (Hungarian, IoU >= 0.5), mean over scenes")
    print(f"{'family':34s} {'components':>10s} {'watershed':>10s} {'merge rule':>10s} {'selection':>10s}")
    for fam in FX.FAMILIES:
        r = {k: [] for k in ("cc", "ws", "mr", "sel")}
        for sd in range(100, 100 + n):
            s = FX.make_scene(fam, sd); P, E, gt = s["P"], s["E"], s["gt"]; ws = MT.ws_psi(P, 0.1)
            r["cc"].append(SC.hungarian(gt, MT.cc(P))["f1"]); r["ws"].append(SC.hungarian(gt, ws)["f1"])
            r["mr"].append(SC.hungarian(gt, CT.merge_rule(ws, P, E))["f1"])
            r["sel"].append(SC.hungarian(gt, MT.select_objects(P, E, sources=CT.psi_candidates(P), **PRM)[0])["f1"])
        print(f"{NAMES[fam]:34s} " + " ".join(f"{np.mean(v):10.2f}" for v in r.values()))


def part2(n, eta=0.01, sigma=6.0):
    tot = {"fixed_cert_px": 0, "fg_px": 0, "fixed_viol": 0, "rigid": 0, "rec_viol": 0, "theory_viol": 0, "band_viol": 0, "runs": 0, "scenes": 0}
    for fam in FX.FAMILIES:
        for sd in range(100, 100 + n):
            s = FX.make_scene(fam, sd); P, E = s["P"], s["E"]; rng = np.random.Generator(np.random.PCG64([7, sd, FX.FAMILIES.index(fam)])); tot["scenes"] += 1
            src = MT.candidates(P); z, rec = MT.select_objects(P, E, sources=src, **PRM); ex, cm = CT.certify_fixed(P, E, PRM, src, rec["S"], eta)
            tot["fixed_cert_px"] += int(cm[z > 0].sum()); tot["fg_px"] += int((z > 0).sum()); pert = CT.adversarial_perturbations(P, E, z, eta, rng)
            for P2, E2 in pert:
                tot["fixed_viol"] += int((cm & (MT.select_objects(P2, E2, sources=src, **PRM)[1]["ident"] != rec["ident"])).sum()); tot["runs"] += 1
            f, ms, c = CT.candidates(P, sigma); ok, H, _ = CT.rigidity(f, ms, eta)
            if ok:
                tot["rigid"] += 1; z2, rec2 = MT.select_objects(P, E, sources=c, **PRM); ex2, cm2 = CT.certify_recomputed(P, E, PRM, H, rec2["S"], z2, eta)
                for P2, E2 in pert:
                    _, _, c3 = CT.candidates(P2, sigma); reg = [[i for i, (_, R) in enumerate(H) if R[y, x]] for y, x in c3]
                    if len(c3) != len(H) or any(len(g) != 1 for g in reg): tot["theory_viol"] += 1; continue
                    src3 = [None] * len(H)
                    for yx, g in zip(c3, reg): src3[g[0]] = yx
                    tot["rec_viol"] += int((cm2 & (MT.select_objects(P2, E2, sources=src3, **PRM)[1]["ident"] != rec2["ident"])).sum())
            lab, cert = CT.threshold_band(P, eta)
            for P2, _ in pert:
                l2 = MT.ccf(P2, 0.7)
                for C, Q in cert.values():
                    ids = np.unique(l2[C])
                    if len(ids) != 1 or ids[0] == 0 or ((l2 == ids[0]) & ~Q).any(): tot["band_viol"] += 1
    print(f"\nPart 2: certificates at eta = {eta} ({tot['runs']} adversarial perturbations over {tot['scenes']} scenes)")
    print(f"  fixed candidates:      certified share of foreground {tot['fixed_cert_px'] / max(tot['fg_px'], 1):.2f}; violations {tot['fixed_viol']}")
    print(f"  recomputed candidates: rigid scenes {tot['rigid']}/{tot['scenes']}; theory violations {tot['theory_viol']}; pixel violations {tot['rec_viol']}")
    print(f"  threshold band:        violations {tot['band_viol']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--scenes", type=int, default=5); a = ap.parse_args()
    part1(a.scenes); part2(a.scenes)
