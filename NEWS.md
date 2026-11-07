# imfeatures 0.1.0 (development)

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
