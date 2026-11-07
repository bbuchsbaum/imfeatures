# Saliency-weighted edge entropy with spatial controls

Filters an image once and weights the resulting edges by mean-normalized
saliency. Returns image structure, map concentration, and control
diagnostics separately. The existing
[`compute_edge_entropy()`](https://bbuchsbaum.github.io/imfeatures/reference/compute_edge_entropy.md)
result layout is unchanged.

## Usage

``` r
compute_saliency_features(
  image,
  saliency,
  representation,
  signed_policy = "reject",
  image_id = if (is.character(image)) image else "image",
  max_pixels = 120000L,
  gabor_bins = 24L,
  filter_length = 31L,
  maxdiag = 500L,
  circ_bins = 48L,
  ranges = list(c(20L, 80L), c(80L, 160L), c(160L, 240L)),
  pairwise = TRUE,
  pair_weighting = "both",
  edge_selection = "image",
  validity = "legacy",
  min_cell_mass_fraction = NULL,
  min_effective_pairs = NULL,
  centre_sigma = 0.25,
  transform_controls = TRUE,
  donor_maps = list(),
  donor_assignments = NULL,
  scales = NULL,
  scale_units = "relative",
  pixels_per_degree = NULL,
  provenance = list(),
  use_cpp = TRUE,
  centre_match = FALSE,
  donor_aspect_tolerance = 0.01,
  min_common_coverage = 0,
  boundary = c("zero", "reflect")
)
```

## Arguments

- image:

  Grayscale numeric matrix or image file path. Matrix axes follow
  [`compute_edge_entropy()`](https://bbuchsbaum.github.io/imfeatures/reference/compute_edge_entropy.md)/imager
  conventions; the map must have the same axes.

- saliency:

  Nonempty numeric matrix of aligned saliency values.

- representation:

  Required: `"weights"` or `"log_density"` (natural logs).

- signed_policy:

  For raw weights: `"reject"`, `"rectify"`, or `"shift"`.

- image_id:

  Target identifier used in donor assignments; defaults to the path for
  file inputs and `"image"` for matrices.

- max_pixels:

  Analysis pixel count for file inputs, as in
  [`compute_edge_entropy()`](https://bbuchsbaum.github.io/imfeatures/reference/compute_edge_entropy.md).
  NULL disables resizing; matrices are not resized.

- gabor_bins, filter_length, maxdiag, circ_bins, ranges:

  Filter and pairwise settings as in
  [`compute_edge_entropy()`](https://bbuchsbaum.github.io/imfeatures/reference/compute_edge_entropy.md).
  Ranges are inclusive distance-index bounds, not a new
  physical-distance convention.

- pairwise:

  Calculate pairwise features and support diagnostics.

- pair_weighting:

  `"both"` for saliency at both endpoints or `"reference"`.

- edge_selection:

  `"image"` holds the selected edges fixed across controls; `"saliency"`
  also changes selection and is a separately labelled sensitivity.

- validity:

  `"legacy"` uses the native legacy mass cutoff (approximately mass
  \> 1) in both backends. `"support"` uses explicit support thresholds.
  Both backends use Euclidean distance geometry.

- min_cell_mass_fraction, min_effective_pairs:

  Required nonnegative support thresholds when validity is `"support"`.
  Cell fraction uses all counted mass.

- centre_sigma:

  Gaussian centre-bias standard deviation as a fraction of each image
  dimension (one or two values). This is an explicit descriptive
  control, not a fitted human fixation model.

- transform_controls:

  Include exact flips and grid-preserving rotations.

- donor_maps:

  Named list of maps, with the same representation and declared
  model/preprocessing as the target.

- donor_assignments:

  Output of
  [`saliency_donor_assignments()`](https://bbuchsbaum.github.io/imfeatures/reference/saliency_donor_assignments.md).
  NULL reports donor controls as unavailable; no random pixel map is
  substituted.

- scales:

  Optional multiscale block sizes. NULL skips multiscale output.

- scale_units, pixels_per_degree:

  Units and conversion for multiscale output; see
  [`saliency_multiscale_entropy()`](https://bbuchsbaum.github.io/imfeatures/reference/saliency_multiscale_entropy.md).

- provenance:

  List recording model/version, smoothing, and other source details
  supplied by the caller. No additional smoothing is applied here.

- use_cpp:

  Use compiled filtering, batched counting and statistics.

- boundary:

  Convolution boundary handling, as in
  [`compute_edge_entropy()`](https://bbuchsbaum.github.io/imfeatures/reference/compute_edge_entropy.md):
  `"zero"` (default, historical) or `"reflect"`, which avoids artificial
  frame edges. Whole-image and foreground-weighted features differ in
  how much weight frame responses receive, so use `"reflect"` when
  comparing them.

- centre_match:

  Also include a centred Gaussian fitted to the observed map's
  `effective_fraction`. This matches location entropy, not shape or pair
  support. Unattainable fits on the discrete grid are explicitly
  unavailable.

- donor_aspect_tolerance:

  Maximum relative aspect-ratio error for donor maps (default 0.01).
  Resizing and before/after location metrics are recorded.

- min_common_coverage:

  Minimum fraction of distance-direction cells shared by all maps in a
  comparison, within each range. Below this fraction, common contrasts
  are NA. Zero requires at least one shared cell; it is not a
  scientifically validated support threshold.

## Value

A list with `features` (one row per map/control), `ranges` (pairwise
range summaries), `cells` (per-control support tables), `multiscale`,
`selection` (selected coordinates per control), `comparisons` (separate
transform and donor family summaries), and `metadata`. Differences named
`observed_minus_control` use each map's own valid cells. Pairwise
differences named `observed_minus_control_common` use their
intersection, with the same equal-directions-then-equal-distances
reduction for both maps. Family comparisons intersect cells across the
observed map and all usable, distinct nonidentity maps in that family,
and report the mean, SD and range of control entropies, coverage,
retained mass, controls used, and status. `n_requested` counts candidate
rows (including unavailable placeholders), `n_available` counts rows
with available descriptors, and `n_controls` counts distinct nonidentity
maps. Repeated donor maps receive one vote. On observed rows,
`observed_minus_transform_mean` and `observed_minus_donor_mean` copy the
corresponding family contrast; other rows have NA. Pairwise family
contrasts always use common support. Entropies are in bits. Zero mass is
NA with an explicit status.

## Details

Pair effective count measures weight concentration, not independent
samples. Weighting changes the image descriptor; concentration need not
decrease entropy. Unweighted differences describe the overall effect of
weighting, while transforms and donors help assess alignment. Donors
preserve their own concentration, not necessarily the target's. Matching
`effective_fraction` alone does not match effective pair support. Low
coverage or effective support calls for inspecting cells, even when a
range value is available. Uniform saliency preserves native legacy
outputs. The historical geometry and validity differences remain only in
`compute_edge_entropy(use_cpp=FALSE)`; the new API agrees across
backends within floating-point tolerance. Controls sharing an edge set
are counted in one geometry pass with one unweighted baseline.
Saliency-based selection can require separate passes. Donor and
transformed-map comparisons are descriptive, not permutation tests.

## Examples

``` r
img <- outer(seq_len(12), seq_len(12), function(x, y) sin(x) + cos(y))
sal <- matrix(1, 12, 12)
result <- compute_saliency_features(img, sal, "weights", pairwise = FALSE,
  filter_length = 5L, gabor_bins = 4L, transform_controls = FALSE)
result$features[, c("control", "first_order", "effective_fraction")]
#>             control first_order effective_fraction
#> 1          observed    1.946194          1.0000000
#> 2        unweighted    1.946194          1.0000000
#> 3       centre_bias    1.989252          0.7782512
#> 4 donor_unavailable          NA                 NA
```
