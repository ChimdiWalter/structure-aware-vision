# Structure-aware vision

Geometry- and topology-aware representation learning for vision. This research asks three questions:
- what foundation-model features encode about geometry;
- how object geometry can be made executable, with certified error bounds;
- when visual context changes what a network considers an object.

Research by Chimdi Walter Ndubuisi, University of Missouri. Manuscripts and code are being added; each part states its current status.

## 1. What vision features encode about geometry
*Status: manuscript under review. Code will be released after review.*

This part measures the local (pullback) geometry of dense foundation-model features (DINOv2), with a method calibrated against analytic ground truth.

**Main finding (preregistered).** The measured geometry changes systematically with how much surrounding context the encoder sees. The change was monotonic in all seven test leaves, so the geometry is not automatically intrinsic to the image. This claim is limited to the representation and tiling tested.

## 2. Executable object geometry
*Status: manuscript in preparation.*

This part develops a Finsler-metric formulation in which a learned field induces geodesic basins, and those basins define the object partition directly.
- **Error bound.** The reference solver has a closed-form error bound for its angular discretisation, sec(θ/2) − 1. On every constant-metric test case, the observed maximum error was 0.97–1.00 of the predicted bound.
- **Differentiable implementation.** It reproduces the reference solver to machine precision.
- **Negative result.** On real leaf images, seeding objects from persistent maxima and assigning pixels by geodesic distance did not outperform inexpensive geodesic or watershed baselines. This result motivated part 3.

## 3. Object identity: context and certificates
*Status: exploratory. Synthetic demos are being added.*

When does changing the visual context change what a network considers a single object? Can an object's existence and its membership be certified under bounded perturbations?
- **Context.** Shifting only the tile grid changed object identity for 44% of a segmentation model's objects on seven validation leaves. Against the current binary annotations, this instability did not detect structural errors better than pixel entropy did.
- **Certificates.** Interval-bounded certificates for object existence and membership were sound on synthetic scenes: there were no violations in 8,464 adversarial perturbation tests. At small perturbations they certified up to 93% of foreground pixels.

## Testbed
The main testbed is plant phenotyping: lesion segmentation on maize leaf photographs. The questions themselves are general.
