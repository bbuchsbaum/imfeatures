# Saliency-modulated entropy specification

Status: implemented through `compute_saliency_features()`,
`saliency_location_entropy()`, `saliency_donor_assignments()`, and
`saliency_multiscale_entropy()`; acceptance tests accompany the implementation.
Date: 2026-09-23. Source reviewed: `18702dcad4ae30d14b7f928a04886b86188aa32c`.

## Purpose and scope

Measure image edge structure emphasized by a saliency map separately from the
distribution and spatial organization of the map itself. The initial deliverables
are saliency-weighted first-order orientation entropy, saliency concentration,
and matched controls. Pairwise orientation entropy follows, then multiscale
location entropy. Level co-occurrence entropy is outside this initial scope.

These are model-weighted image descriptors. Controls do not establish an effect
of human attention; that interpretation requires independent fixation evidence.

## Inputs and preprocessing

- Filter the original image once. Saliency weights the resulting responses;
  never multiply the image by the map before filtering.
- Let `a_i` be nonnegative edge strength and `theta_i` the winning orientation
  bin after existing border handling. Require finite responses. Materially
  negative responses are invalid for these probability calculations; do not
  silently take absolute values or change the existing filter definition.
- Require an explicit map representation: nonnegative weights or natural-log
  weights/density. Do not infer it from the observed value range.
- For nonnegative weights, require finite values and positive total mass.
  For log input, allow finite values and `-Inf` (zero mass), require at least one
  finite value, and compute `q_i = exp(l_i - logsumexp(l))`. Reject `NA`, `NaN`,
  and `+Inf`. Negative log values are valid and must not be clipped.
- Signed raw maps default to rejection. Explicit alternatives are rectification
  (`max(s_i, 0)`) or shifting (`s_i - min(s)` when the minimum is negative).
  Record the policy; neither transformation establishes fixation calibration.
  These policies apply only to raw weights. No implicit pseudocount, gamma,
  temperature, per-image rank transform, or min-max normalization is applied.
- All-zero maps, including all-`-Inf` log maps or maps made zero by rectification,
  have unavailable descriptors with a `zero_map_mass` status. Malformed values
  or incompatible dimensions are input errors. Do not substitute a uniform map.
- Initially require a numeric map aligned to the image's original pixel grid.
  Apply any image resize/crop to the map using the same spatial transform;
  convert log input to positive mass/density before interpolation. Use area averaging when downsampling and linear interpolation when upsampling;
  shrink first for mixed-axis resizes and renormalize mass. Record source/analysis
  dimensions, interpolation, and observed/donor location metrics before and after
  resizing. No implicit registration. Donor aspect ratios allow a declared
  relative tolerance (default 0.01) for small grid-rounding differences.
- `q_i = s_i / sum(s)` is defined over the full analysis image, including areas
  without edges. Edge normalization below uses a different domain.
- Record model/version, input representation, any existing model smoothing,
  and additional smoothing. Do not add smoothing by default. When used, specify
  visual degrees if viewing geometry is known, otherwise image-relative units;
  retain the pixel conversion. Keep this policy common to targets and controls.

## Edge selection and saliency normalization

First order uses all positive responses after border handling (`E1`), as in the
existing pre-threshold histogram. Pairwise counting uses a separate set `E2`.

The default pair selection ranks by image response `a_i`. Reproduce the existing
cutoff: the 10,000th largest response when the image has at least 10,000 pixels;
otherwise the `ceiling(0.1 * N)`th, with a minimum rank of one. Threshold using
the full response array, including zeros; retain ties and then exclude zeros.
This is a rank-based threshold, not an exact edge-count cap. Reuse the identical
`E2` across the original map and every control for that image.

An explicit sensitivity mode may rank by `s_i * a_i` using the same rank rule.
Label it separately: it changes both selection and weighting, and cannot claim
to isolate reweighting. Do not replace the default results with it.

Within the edge set for the relevant order, define

```
mean_s(E) = sum(s_i for i in E) / length(E)
t_i      = s_i / mean_s(E)
```

Use `E1` for first order and `E2` for pairs. The mean is over selected edge
locations, including those whose saliency is zero; it is not response-weighted.
For numerical stability, first scale the relevant saliency values by their
maximum, then divide by their mean. A positive constant map on the relevant
edge set must take an explicit all-ones path. This normalization is invariant
to positive rescaling of the supplied weights and recovers unit weights for a
uniform map. Finite arithmetic may require a tolerance for nonuniform maps.

If the edge set is empty, return `NA` with `no_edges`. If it exists but contains
no positive saliency, return `NA` with `zero_edge_saliency`. Do not divide by zero
or fall back to an unweighted result. Full-image saliency descriptors can remain
available when an edge descriptor is unavailable.

Normalization preserves relative saliency on edges; it discards absolute map
mass there. Also report `sum(q_i for i in E1)` and `sum(q_i for i in E2)` so
saliency concentrated in smooth, excluded, or border regions remains visible.

## Measurements

### First-order orientation entropy

```
v_i = a_i * t_i
C_k = sum(v_i * indicator(theta_i == k), i in E1)
p_k = C_k / sum(C)
H1  = -sum(p_k * log2(p_k), p_k > 0)
```

Return bits and retain the original orientation bins. Zero histogram mass is
unavailable, not zero entropy. A positive-mass histogram with one occupied
orientation has entropy zero.

### Saliency location concentration

For the full-image probability mass `q` on `N` equal-area pixels, report

```
H_location         = -sum(q_i * log2(q_i), q_i > 0)
KL_uniform         = log2(N) - H_location
effective_fraction = 2^(-KL_uniform)
```

Uniform mass gives `(log2(N), 0, 1)`; a one-pixel point mass gives
`(0, log2(N), 1/N)`. KL and effective fraction are invariant to subdividing every
pixel into the same number of equal-mass subpixels. This is not a guarantee of
invariance to arbitrary interpolation, blur, cropping, or changes in field of
view. Do not substitute `H_location / log2(N)` as a resolution-invariant measure.

### Pairwise orientation entropy

Default to `w_ij = a_i * a_j * t_i * t_j`. An explicitly named reference-only
variant uses `w_ij = a_i * a_j * t_i`. Pairs are ordered: direction is defined
relative to the reference edge's orientation even with symmetric weights.

Use correct Euclidean distance in both new backends. Preserve native distance
clipping, direction/relative-orientation bins, ordered pairs including self-pairs,
and range-index conventions. The historical R off-by-one geometry remains only
in the legacy API and the explicit fixed-response test switch.

Normalize orientation counts within each `(distance, direction)` cell. Preserve
the current summary: mean over valid directions at each distance, then mean
over valid distances in the requested range. This is not a pair-mass-weighted
mean. Return `NA` when a range contains no valid cells.

#### Compatibility and validity

The default validity policy is `legacy`. Mean-normalized saliency makes it
invariant to saliency amplitude while preserving the existing absolute cutoff.
It does not remove sensitivity to image contrast, filter amplitude, or the
chosen edge set. The compatibility path retains existing arithmetic, including
the denominator offset; that offset is not an orientation-bin pseudocount.

The new API uses the native legacy predicate in both backends: `(M_cell +
1e-5) - 1e-5 > 1`, preserving its finite-arithmetic boundary and C++ uniform-map
results. Historical R validity (`M_cell + 1e-5 > 1`) and displaced geometry
(`sqrt((dx-1)^2 + (dy-1)^2)`) remain in `compute_edge_entropy(use_cpp = FALSE)`.
They are not inherited by this new API. Metadata records `euclidean` geometry.
Actual image-to-feature tests check filtering, selection, geometry, support and
summaries across R and C++, including reference-only weighting and saliency-based
selection. Floating-point accumulation differences use explicit tolerances.

An opt-in `support` policy uses positive cell mass plus caller-specified minimum
cell mass fraction and minimum effective pair count. Define the mass fraction
relative to total counted pair mass over the whole count cube, before validity
filtering. Require explicit thresholds for this research mode; do not assert a
scientifically validated default. Its R/C++ predicate is identical and does not
use the legacy offset. Empty cells remain unavailable under every policy.

Uniform positive maps must reproduce the native backend's existing finite,
defined outputs in legacy mode, preserving reduction order or using the
existing unweighted path. This guarantee does not include preserving the
existing C++ zero-mass value of zero: correct that case to `NA`, matching R,
with an explicit release note and regression test. Do not change valid zero
entropy from a one-orientation positive-mass histogram. Calls without saliency
otherwise retain their existing results and output structure.

The file-input regression test exposed an additional conversion defect:
`as.matrix()` flattened imager's four-dimensional grayscale array into one
column before the legacy C++ call. Preserve the two spatial dimensions
explicitly. This intentionally corrects the old file path's spurious zero/NA
results; ordinary matrix-input values remain unchanged. Compare file inputs
with the corresponding correctly shaped grayscale matrix in the regression.

#### Support diagnostics

For each cell, accumulate positive pair count, occupied orientation-bin count,
`M = sum(w)`, `Q = sum(w^2)`, and `n_eff = M^2 / Q` when `Q > 0`. Accumulate the
squared individual pair weights, not squared orientation-bin totals. Compute
the same diagnostics with unweighted pair strengths `a_i * a_j` on the same
edge set. Report both effective counts and their ratio where defined.

At each distance range, return pair-mass fraction, effective pair count from
the range's summed `M` and `Q`, valid-cell coverage, fraction of mass retained
by validity filtering, and the unweighted counterparts. Overlapping ranges
are separate summaries, so their mass fractions need not sum to one. Preserve
per-cell diagnostics for inspection; a range average does not certify cell
support. Mark zero denominators unavailable rather than substituting zero.

Effective pair count describes concentration of weights, not independence or
an entropy ceiling. Pairs share endpoints. Only the number of occupied
orientation bins gives the categorical bound `H <= log2(K_occupied)`; three
actual positive pairs occupy at most three bins, but `n_eff = 3` can coexist
with many occupied bins and entropy above `log2(3)`. Small fractional mass alone
also does not establish small support. Do not attach an iid bias correction or
standard error to these deterministic descriptors without a sampling model.

## Controls

Compute controls alongside each weighted edge measure using fixed image
responses, default edge selection, bins, validity policy, and preprocessing.
Each control receives its own selected-edge mean normalization. Always include
the unweighted result and an image-independent centre-bias map specified in
normalized image coordinates. Record the source and parameters of that map;
if estimated from fixations, use a separate training/reference set.

Use two separately labelled spatial control families:

1. **Donor-map controls:** randomly permute map-to-image assignments without
   assigning an image its own map, within declared aspect-ratio/model/processing
   strata. Exclude duplicate versions of the target stimulus. Record seed,
   donor identifiers, strata, and requested/available replicate counts. Prefer
   equal analysis dimensions; record any required resize. If the eligible pool
   is empty, report unavailable controls rather than substituting pixel noise.
   Donors preserve their own map structure, not the target map's concentration.
   Resizing can change both smoothness and concentration.
2. **Within-map transformations:** use nonidentity horizontal/vertical flips
   and 180-degree rotation; 90/270-degree rotations are admissible on square
   grids. Use exact grid transformations without interpolation. These preserve
   the value multiset, location entropy, and spatial structure up to the
   transformation. They change alignment with the image, but need not destroy
   it: symmetric maps can be unchanged. Record identity-equivalent outcomes
   and duplicate transformations; do not count them as independent null draws.

Pixel permutation is excluded from the primary controls because it destroys
spatial smoothness and clustering. It actually preserves the value histogram,
location entropy, and effective fraction; do not claim otherwise. Arbitrary
angle rotations, cropping, and interpolated warps do not have the exact
preservation properties of the grid transformations above.

Return individual control values, support diagnostics, and matched differences,
not only a control mean. Label donor and transformed-map comparisons separately.
They are descriptive controls; inferential permutation p-values would require
an explicit exchangeability argument and enough distinct controls.

The implementation uses a seeded randomized bipartite matching within each
stratum, restoring the caller's RNG state. It is not a uniform sampler over all
admissible permutations. The default centre control is a Gaussian with standard
deviation 0.25 of each image dimension; this parameter is exposed and recorded,
not asserted to be a calibrated fixation model.

### Interpretation and common support

Observed-minus-unweighted describes the overall effect of weighting, not an
alignment-specific effect. Weighting can raise or lower entropy and effective
support. For example, two differently oriented edges of strengths 3 and 1 have
ordered-pair relative-orientation counts (10,3,3) when all pairs share one cell.
Saliency weights proportional to (1,3) make edge strengths equal: entropy rises
from 1.329434 to 1.5 bits and effective pair count from 2.56 to 4. This deterministic
descriptor has changed its target distribution; it is not a biased estimate of
the original unweighted entropy.

Keep individual `observed_minus_control` differences over each map's own valid
cells. Also return `observed_minus_control_common` after intersecting observed
and control validity within each range. Both use identical cells and identical
aggregation weights (equal directions within each distance, then equal retained
distances). Report shared cell coverage, distance coverage, and retained mass
for each map. No shared cells, or coverage below the explicit
`min_common_coverage` threshold, gives `insufficient_common_support` and NA.
The default threshold zero requires at least one shared cell; it does not
assert a scientifically validated minimum.

`comparisons` returns separate transform and donor summaries. For each family
and metric/range, retain controls with an available descriptor, remove maps
identical to the observed map, and deduplicate within that family. Report all
candidate/available/distinct counts and the control IDs used. Family pairwise
comparisons intersect validity across the observed map and all retained maps
before calculating any entropy mean. Return observed entropy, mean/SD/min/max of
control entropies, their difference, and common-support diagnostics. SD is
control dispersion, not a standard error. Do not pool families or histogram
counts. Repeated donor assignments receive one vote per distinct map.

On observed feature/range rows, add `observed_minus_transform_mean` and
`observed_minus_donor_mean`; other rows have NA. Donors are not automatically
concentration-matched: use a declared concentration stratum if required and
inspect the effective fractions after resizing.

With `centre_match = TRUE`, retain the fixed centre control and add a centred
Gaussian fitted to the observed effective fraction. This fits one sigma shared
by the normalized image axes. Record target and achieved fractions, sigma, and
fit status. The uniform limit has infinite sigma. Fractions below the discrete
Gaussian family's minimum are unavailable (`unattainable_effective_fraction`),
not silently approximated. Matching location entropy does not match map shape,
pair-distance mass, or effective pair support.

### Counting cost

Group controls by their selected edge set. Compute each pair's geometry once
per group and update all control histograms in that pass; accumulate the
unweighted counts and diagnostics once per group. Reuse baseline statistics in
all corresponding summaries. The default image-based selection has one group.
Native histograms interleave controls for adjacent memory access. A slow batched
R reference and hand-enumerated single-map fixtures check the actual native
outputs. Runtime remains quadratic in selected edges and increases with the
number of controls; benchmark representative images and report workload details.

## Multiscale location entropy

For each declared block size `b` and grid origin, sum pixel probability mass
within each intersected block `B`: `q_B = sum(q_i, i in B)`. Compute
`H(b) = -sum(q_B log2(q_B))`. Use several predefined shifted origins and return
the full curve with origin-specific values and an origin-averaged summary.

Clip boundary blocks to the image; do not wrap, duplicate, or discard mass.
For a uniform-reference comparison at each scale, use
`u_B = area(B) / area(image)` and `KL_b = sum(q_B log2(q_B/u_B))`. The shortcut
`log2(number_of_blocks) - H(b)` applies only to equal-area blocks. Define block
sizes in visual degrees when available, otherwise image-relative units, and
record their pixel realization and origins.

Call this multiscale location entropy, not second-order entropy. It can reveal
scale-dependent dispersion, but does not uniquely identify hotspot number or
arrangement. A slope is an optional derived summary with its fit range stated;
retain the curve. Do not assume monotonicity across nonnested shifted grids.

## Results and implementation order

New results must identify metric, image/control, distance range or spatial scale,
value, units, availability reason, selection/weighting/validity policies, and
preprocessing provenance. Preserve saliency descriptors separately from edge
descriptors. Keep `complex_before` as the existing unweighted image quantity.
Store pairwise cell diagnostics separately from compact feature summaries.

1. Add first-order weighting, location concentration, input validation, explicit
   zero-mass handling, and the controls. Preserve existing ordinary outputs.
2. Add pairwise weighting, selected-edge normalization, legacy and opt-in support
   validity, support diagnostics, and selection sensitivity fixtures.
3. Add the multiscale curve and shifted-partition diagnostics.

Implement both R and actual compiled C++ paths where the existing edge routines
have both backends. Do not replace compiled-path validation with mocks. Changes
to unrelated edge binning or filters are outside this specification.

## Acceptance evidence

- Fixed response fixtures: one occupied orientation gives 0 bits; two equally
  weighted orientations give 1 bit; weights in a 3:1 ratio give
  `0.811278124459` bits. Use hand-enumerated pair positions/orientations to verify
  ordered reference-only versus symmetric weighting and count-cube placement.
- Uniform saliency reproduces defined native legacy results and agrees across
  both new backends within tolerance. Test
  several constant amplitudes and preserve unsupported cells. No-saliency
  regression fixtures retain all defined values and output columns.
- Positive rescaling leaves entropy, validity, effective support, mass fractions,
  selection, and location descriptors unchanged within numerical tolerance.
  Adding a constant to log weights has the same invariance.
- Zero map mass, no edges, zero saliency on selected edges, empty pair cells,
  and unsupported ranges yield the specified unavailable states. A single
  occupied orientation with positive mass remains a valid zero. Verify both
  direct entropy helpers and public R/C++ paths, including the documented
  correction to the old C++ zero-mass result.
- Selection fixtures include a weak salient edge excluded by image ranking but
  retained by saliency ranking, cutoff ties, and the small-image rank rule.
  Default control runs must retain identical selected coordinates.
- Validity fixtures bracket mass 1 and the R/C++ legacy boundary interval.
  Assert the shared native predicate and cross-backend agreement in both modes.
  Keep historical R behavior confined to the legacy API.
- Support fixtures distinguish small mass from concentrated weights, compare
  weighted/unweighted support, and distinguish sums of squared pair weights
  from squared bin totals. Include the effective-count counterexample below.
- Location fixtures include uniform/point mass, equal-mass grid subdivision,
  equivalent weight/log inputs, and all signed-input policies. Include an
  asymmetric spatial fixture to catch transpose and resize-alignment mistakes.
- Control fixtures verify excluded self-donors, reproducible assignments,
  exact transformation preservation, symmetric-map degeneracy, and reporting
  of resize-induced changes rather than claiming exact invariance.
- Multiscale fixtures enumerate block masses by hand, including partial boundary
  blocks and shifted origins; compare a contiguous hotspot and separated mass
  with the same fine-grid value histogram.
- For small fixed response fixtures, use approximately `1e-12` absolute
  tolerance in bits; compare categorical masks and statuses exactly. For
  image-to-feature parity retain the existing `5e-4` test tolerance initially,
  documenting discrepancies rather than broadening it. Add no performance or
  human-attention claims without separate evidence.

### Checks completed while drafting

Independent Python arithmetic verified selected-edge mean-normalization under
128-fold scaling and KL invariance under four-way equal-mass subdivision for
uniform and point-mass fixtures. It also evaluated a 24-orientation distribution
with one weight `p = (1 + sqrt(161))/24` and 23 weights `(1-p)/23`:

```
n_eff       = 3.000000000000
H           = 2.929184265652 bits
log2(n_eff) = 1.584962500721 bits
```

This refutes an entropy ceiling based on effective count. These are specification
checks, not evidence that the proposed R/C++ features are implemented or tested.

References: [weighted effective count](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html),
[DeepGaze input/output examples](https://github.com/matthias-k/DeepGaze).
