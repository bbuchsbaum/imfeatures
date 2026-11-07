# imfeatures 0.1.0 (development)

- Add `boundary = c("zero", "reflect")` to `compute_edge_entropy()` and
  `compute_saliency_features()`. The native filter bank zero-pads the image,
  so the frame acts as a strong straight edge; only a 2-pixel border was
  discarded, and frame responses 3-4 pixels deep could enter the retained
  edge set. On full-frame photographs this barely changes features, but on
  objects shown against plain backgrounds it can substantially change pairwise
  entropy, especially `pentropy_160_240`. `"reflect"` mirrors the image at its
  edges and removes the artifact; it is recommended for new analyses. The
  default stays `"zero"` so earlier results are reproduced exactly.

- Use OpenCLIP validation preprocessing in `clip_features()` instead of random
  training augmentation when extracting image features (#51).
- **Recompute affected file-input edge entropy analyses:** file paths passed to
  `compute_edge_entropy()` with `use_cpp = TRUE` (the default) were flattened
  into one column by an array-to-matrix conversion. Both spatial dimensions are
  now preserved. Previous results from that path are unreliable; matrix inputs
  were unaffected by this conversion defect.
- Add `compute_saliency_features()` for saliency-weighted first-order and
  pairwise edge entropy, with selected-edge mean normalization, support
  diagnostics, centre-bias controls, exact map transformations, and donor maps.
- Add `saliency_location_entropy()`, `saliency_multiscale_entropy()`, and
  reproducible `saliency_donor_assignments()`. Map representation is explicit;
  zero mass remains distinct from a valid entropy of zero.
- Correct native edge entropy to return `NA` for zero histogram mass, matching
  R. Defined matrix-input legacy outputs and the existing result layout are preserved.
- Count saliency controls sharing an edge set in one geometry pass, and reuse
  baseline counts and support statistics. Both new backends use Euclidean
  geometry and the native legacy validity rule. Historical R conventions remain
  confined to `compute_edge_entropy(use_cpp = FALSE)`.
- Add common-support pairwise contrasts, separate means for distinct transform
  and donor controls, and an optional entropy-matched Gaussian centre control.
  Unweighted contrasts describe the overall effect of weighting; matched
  comparisons help distinguish spatial alignment from concentration effects.
- Average probability maps when downsampling, record observed and donor
  concentration before/after resizing, and allow a declared small donor
  aspect-ratio tolerance. Add a runnable saliency vignette and localization,
  backend-parity, common-support, and realistic-size regression tests.
