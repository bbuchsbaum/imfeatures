# imfeatures 0.1.0 (development)

## Spatial pooling: documentation, a new vignette, and three fixes

Convolutional feature maps can be reduced over space in several ways: global
average or max, a coarse grid, low-frequency DCT modes, centre-to-periphery
rings, or not at all. These options are now documented in one place.

- New vignette, `vignette("spatial-pooling")`. It shows what each option
  keeps, how wide the resulting VGG-16 features are, and how the string
  options (`"avg"`, `"max"`, `"none"`, `"resize_HxW"`) relate to the pooler
  objects (`global_average_pool()`, `adaptive_grid_pool()`, `dct_pool()`,
  `radial_pool()`, `identity_pool()`). It runs without Keras or Python.
- New help page `?poolers` compares the pooler objects. The help for each
  pooler, `pool_activations()`, `pooler_feature_names()`,
  `vgg_hierarchy_spec()`, `vgg16_block_end_spec()`, `extract_features()` and
  `extract_vgg_features()` now states output sizes, column order and scale,
  with examples.
- Printing a `vgg_hierarchy_spec` now lists the pooler of each stage.

Fixes:

- `adaptive_grid_pool()` failed with "invalid number of intervals" when
  either grid dimension was 1, for example `adaptive_grid_pool(1, 3)`. A
  single bin now works; `adaptive_grid_pool(1, 1)` equals global average
  pooling. Results for two or more bins are unchanged.
- `radial_pool()` accepted breaks outside `[0, 1]` and silently returned
  all-zero columns for the rings they defined. Such breaks are now an
  error. A ring that holds no positions on a small map still gives zeros,
  now with a warning.
- `extract_vgg_features(pooling = )` accepted only `"resize_3x3"`,
  `"resize_5x5"` and `"resize_7x7"`, while `extract_features()` accepted
  any `"resize_HxW"`. Both now accept any positive `H` and `W`, and both
  reject a zero size such as `"resize_0x3"` up front.

One naming change: when `extract_feature_hierarchy()` flattens a
convolutional map with `identity_pool()`, columns are now named by channel
and position (`b5::channel_001::y1_x1`) instead of by index
(`b5::f_001`). When a dense stage is given a spatial pooler, its columns
are named `fc1::f_001`, as they pass through unpooled. Values and column
order are unchanged in both cases.

One thing to know: `dct_pool()` returns the spatial mean as its first
column, but its other columns grow with the size of the map (16 times
larger at 224 x 224 than at 14 x 14 for the same pattern). This has not
changed. Standardize DCT columns before combining them; `block_pca()` does
so by default.

## Edge entropy: image-frame artifact and new `boundary` option

**What was wrong.** Edge entropy filters each image with 31 x 31 Gabor
filters. Pixels outside the image were treated as zero (black), so the image
frame looked like a strong, perfectly straight edge at 0 and 90 degrees. Only
a 2-pixel border of filter responses was discarded, but the frame response
reaches 3-4 pixels into the image (about 35% of the border brightness at
pixel 3). Those responses can enter the strongest-edge set used for the
pairwise statistics. Both backends (`use_cpp = TRUE` and `FALSE`) behaved this
way. Shikang raised the concern.

**Who is affected.**

- Full-frame natural photographs: changes are small. On 17 test photographs,
  rank correlations between the old and corrected features were 0.99 or
  higher.
- Objects on plain or uniform backgrounds, or images with large flat regions:
  changes can be large. Frame responses made up 7-20% of the retained edges.
  Rank correlations for `pentropy_160_240` fell to 0.15-0.79, with shifts of
  up to 1.7 bits. Some long-range values exist only because of the frame and
  become `NA` once it is removed.
- Whole-image versus foreground comparisons (for example with
  `compute_saliency_features()`): only the whole image includes the frame
  edges, so part of any difference can be the artifact.

**What changed.** `compute_edge_entropy()` and `compute_saliency_features()`
gain `boundary = c("zero", "reflect")`. `"reflect"` mirrors the image across
its edges before filtering, so the frame creates no edge. Responses more than
15 pixels from the edge are unchanged. The default stays `"zero"`, so
existing scripts reproduce earlier results exactly. Use `"reflect"` for new
analyses.

**How to run it.**

```r
# Install this version
remotes::install_github("bbuchsbaum/imfeatures", ref = "claude/gallant-meitner-a4r2x3")
library(imfeatures)

compute_edge_entropy("stim/img001.jpg", boundary = "reflect")
compute_saliency_features("stim/img001.jpg", sal_map, "weights", boundary = "reflect")
```

**How to tell whether your results change.** Compute the features both
ways on your actual stimuli and compare.

```r
files <- list.files("stim", pattern = "\\.(jpe?g|png)$", full.names = TRUE)
feats <- c("entropy", "pentropy_20_80", "pentropy_80_160",
           "pentropy_160_240", "complex_before")
old <- do.call(rbind, lapply(files, compute_edge_entropy, boundary = "zero"))
new <- do.call(rbind, lapply(files, compute_edge_entropy, boundary = "reflect"))

# 1. Do image rankings change? Spearman rho near 1 means the same order.
sapply(feats, function(f) cor(old[[f]], new[[f]], method = "spearman",
                              use = "complete.obs"))

# 2. How big are the changes, relative to the spread across images?
sapply(feats, function(f) {
  d <- new[[f]] - old[[f]]
  c(mean_diff = mean(d, na.rm = TRUE),
    max_abs_diff = max(abs(d), na.rm = TRUE),
    sd_across_images = sd(old[[f]], na.rm = TRUE))
})

# 3. Which values only existed because of the frame?
sapply(feats, function(f) sum(!is.na(old[[f]]) & is.na(new[[f]])))

# 4. Which images moved most? Look at these first.
d <- abs(new$pentropy_160_240 - old$pentropy_160_240)
head(old$im[order(d, decreasing = TRUE)], 10)
```

How to read the comparison:

- If every Spearman rho is about 0.98 or higher, and the largest change is
  small next to the spread across images, the artifact is unlikely to matter.
  Still rerun the main models once to confirm.
- With few images (under about 20), rho can drop because of very small
  changes that happen to swap close neighbours. In that case judge mainly by
  `max_abs_diff` against `sd_across_images`.
- If rho is lower for any feature, or values become `NA`, rerun the
  downstream analyses with `"reflect"` features and compare the effect
  sizes and conclusions, not only whether results stay significant. Long
  distance ranges such as `pentropy_160_240` are the most likely to change.
- For whole-image versus foreground hypotheses, compute both measures with
  `boundary = "reflect"` before interpreting the difference.
- To report results either way, state which `boundary` setting was used.
  `compute_saliency_features()` records it in `metadata$filter$boundary`.

## Other changes

- Fix `compute_feature_similarity()` with `lowmem = TRUE` (the default). It
  read the image-path column of `extract_features()` output instead of the
  features, so off-diagonal similarities were `NA`, and it left the diagonal
  at 0. It now matches `lowmem = FALSE`.
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
