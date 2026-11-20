# Certified object identity: the theory

Everything here is elementary and proved in a few lines. It builds on geodesic Voronoi partitions, facility location, interval bounds for shortest paths and the stability theorem for persistence diagrams (Cohen-Steiner, Edelsbrunner and Harer, 2007). **No single lemma is claimed as new.** What is studied is whether these pieces, executed and certified jointly, give useful object identities.

## 1. Object formation as facility location
**Setting.**
- Pixels form an 8-neighbour grid graph.
- Evidence fields: P for foreground and E for boundaries.
- Edge weights: w = l (eps0 + rho a^2 + gamma b^2), where a = 1 - (mean P on the edge) and b = mean E.
- Candidates s_1, ..., s_K are persistence maxima of a proposal field.
- Costs: c_i(x) = d_w(s_i, x) for object i, and c_0(x) = kappa P(x) for the background.

**The operator.**
- **Existence:** S* = argmin over S of [ sum over x of min over i in S + {0} of c_i(x) ] + lambda |S|.
- **Membership:** z(x) = argmin over i in S* + {0} of c_i(x).

**How it behaves.**
- A candidate becomes an object only if it pays for its opening cost.
- A second maximum inside one object does not pay, because its region is cheap to reach.
- Touching objects separated by boundary evidence do pay.
- Weak regions lose to the background.

**Lemma 1 (geodesic star-shapedness).** If the background cost is 1-Lipschitz in w, every object region contains its source and is connected. Without that assumption, a region may bridge fragments. Connectedness and bridging cannot both be guaranteed.

## 2. Certificates with fixed candidates
**Lemma 2 (interval geodesics).** If w_lo <= w~ <= w_hi on every edge, then d_{w_lo} <= d_{w~} <= d_{w_hi}. Bounds on the fields of |dP|, |dE| <= eta give exact per-edge w_lo and w_hi.

**Proposition 3 (membership).** For a fixed S, a pixel keeps its label a if c_a^hi(x) < min over j != a of c_j^lo(x).

**Proposition 4 (paired existence).** S* stays the unique optimum if, for every T != S*, sum over x of u_T(x) + lambda (|S*| - |T|) < 0, where u_T(x) is defined as follows:
- u_T(x) = 0 wherever a label shared by S* and T provably wins over S* + T + {0};
- otherwise u_T(x) = min over S* of c^hi(x), minus min over T of c^lo(x).

**Why pairing matters.** It bounds the *difference* of the two objectives within one perturbed realization. The naive bound treats both sides independently, so its width grows with image area and it certifies almost nothing.

## 3. Certificates with recomputed candidates
**Setup.**
- The proposal field is f = G_sigma * P. A Gaussian blur is non-expansive in the sup norm.
- For a maximum m with persistence p, its basin B_m is the component of {f > f(m) - p} that contains m.
- Its localization region is L_m = B_m intersected with {f >= f(m) - 2 eta}.

**Lemma B1 (localized survival).** If p > 2 eta, the maximizer of f~ over B_m is a local maximum of f~. It lies in L_m and has persistence at least p - 2 eta.

**Proposition B2 (rigid candidate set).** Suppose these conditions hold:
- no maximum has persistence in [tau - 2 eta, tau + 2 eta];
- robust maxima are at least eta away from the value threshold;
- the localization regions L_m of the robust maxima are pairwise disjoint.

Then, under every perturbation with ||P~ - P|| <= eta, the recomputed candidates are exactly one point in each L_m and nothing else. The proof combines Lemma B1 with the bottleneck stability of persistence diagrams.

**Proposition B3 (uncertain sources).** A source moving within L_m has cost in

  [ min over s in L_m of d_lo(s, x) ,  d_hi(m, x) + max over s in L_m of d_hi(m, s) ].

**Theorem B4 (whole-pipeline certificate).** Under B2, the certificates of section 2 computed with the intervals of B3 hold when the entire pipeline is re-run, including candidate generation.

## What is not claimed
- **Configurations where candidates can be born or die.** The certificate abstains on them; this is open.
- **Perturbations of the input image**, as opposed to the evidence fields.
- **Global topology** of the regions.
- **Correctness.** A certified answer can be stably wrong. On evidence-identical scene pairs, a stable output is necessarily wrong on one of them.
