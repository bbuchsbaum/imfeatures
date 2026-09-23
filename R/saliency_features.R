#' Saliency-weighted edge entropy with spatial controls
#'
#' Filters an image once and weights the resulting edges by mean-normalized
#' saliency. Returns image structure, map concentration, and control diagnostics
#' separately. The existing `compute_edge_entropy()` result layout is unchanged.
#' @md
#' @param image Grayscale numeric matrix or image file path. Matrix axes follow
#'   `compute_edge_entropy()`/imager conventions; the map must have the same axes.
#' @inheritParams saliency_location_entropy
#' @param image_id Target identifier used in donor assignments; defaults to the
#'   path for file inputs and `"image"` for matrices.
#' @param max_pixels Analysis pixel count for file inputs, as in
#'   `compute_edge_entropy()`. NULL disables resizing; matrices are not resized.
#' @param gabor_bins,filter_length,maxdiag,circ_bins,ranges Filter and pairwise
#'   settings as in `compute_edge_entropy()`. Ranges are inclusive distance-index
#'   bounds, not a new physical-distance convention.
#' @param pairwise Calculate pairwise features and support diagnostics.
#' @param pair_weighting `"both"` for saliency at both endpoints or `"reference"`.
#' @param edge_selection `"image"` holds the selected edges fixed across controls;
#'   `"saliency"` also changes selection and is a separately labelled sensitivity.
#' @param validity `"legacy"` uses the native legacy mass cutoff (approximately
#'   mass > 1) in both backends. `"support"` uses explicit support thresholds.
#'   Both backends use Euclidean distance geometry.
#' @param min_cell_mass_fraction,min_effective_pairs Required nonnegative support
#'   thresholds when validity is `"support"`. Cell fraction uses all counted mass.
#' @param centre_sigma Gaussian centre-bias standard deviation as a fraction of
#'   each image dimension (one or two values). This is an explicit descriptive
#'   control, not a fitted human fixation model.
#' @param centre_match Also include a centred Gaussian fitted to the observed
#'   map's `effective_fraction`. This matches location entropy, not shape or pair
#'   support. Unattainable fits on the discrete grid are explicitly unavailable.
#' @param transform_controls Include exact flips and grid-preserving rotations.
#' @param donor_maps Named list of maps, with the same representation and
#'   declared model/preprocessing as the target.
#' @param donor_assignments Output of `saliency_donor_assignments()`. NULL reports
#'   donor controls as unavailable; no random pixel map is substituted.
#' @param donor_aspect_tolerance Maximum relative aspect-ratio error for donor
#'   maps (default 0.01). Resizing and before/after location metrics are recorded.
#' @param min_common_coverage Minimum fraction of distance-direction cells shared
#'   by all maps in a comparison, within each range. Below this fraction, common
#'   contrasts are NA. Zero requires at least one shared cell; it is not a
#'   scientifically validated support threshold.
#' @param scales Optional multiscale block sizes. NULL skips multiscale output.
#' @param scale_units,pixels_per_degree Units and conversion for multiscale output;
#'   see `saliency_multiscale_entropy()`.
#' @param provenance List recording model/version, smoothing, and other source
#'   details supplied by the caller. No additional smoothing is applied here.
#' @param use_cpp Use compiled filtering, batched counting and statistics.
#' @return A list with `features` (one row per map/control), `ranges` (pairwise
#'   range summaries), `cells` (per-control support tables), `multiscale`,
#'   `selection` (selected coordinates per control), `comparisons` (separate
#'   transform and donor family summaries), and `metadata`. Differences named
#'   `observed_minus_control` use each map's own valid cells. Pairwise differences
#'   named `observed_minus_control_common` use their intersection, with the same
#'   equal-directions-then-equal-distances reduction for both maps.
#'   Family comparisons intersect cells across the observed map and all usable,
#'   distinct nonidentity maps in that family, and report the mean, SD and range
#'   of control entropies, coverage, retained mass, controls used, and status.
#'   `n_requested` counts candidate rows (including unavailable placeholders),
#'   `n_available` counts rows with available descriptors, and `n_controls` counts
#'   distinct nonidentity maps. Repeated donor maps receive one vote.
#'   On observed rows, `observed_minus_transform_mean` and
#'   `observed_minus_donor_mean` copy the corresponding family contrast; other
#'   rows have NA. Pairwise family contrasts always use common support.
#'   Entropies are in bits. Zero mass is NA with an explicit status.
#' @details
#' Pair effective count measures weight concentration, not independent samples.
#' Weighting changes the image descriptor; concentration need not decrease
#' entropy. Unweighted differences describe the overall effect of weighting,
#' while transforms and donors help assess alignment. Donors preserve their own
#' concentration, not necessarily the target's. Matching `effective_fraction`
#' alone does not match effective pair support. Low coverage or effective support
#' calls for inspecting cells, even when a range value is available.
#' Uniform saliency preserves native legacy outputs. The historical geometry
#' and validity differences remain only in `compute_edge_entropy(use_cpp=FALSE)`;
#' the new API agrees across backends within floating-point tolerance.
#' Controls sharing an edge set are counted in one geometry pass with one
#' unweighted baseline. Saliency-based selection can require separate passes.
#' Donor and transformed-map comparisons are descriptive, not permutation tests.
#' @examples
#' img <- outer(seq_len(12), seq_len(12), function(x, y) sin(x) + cos(y))
#' sal <- matrix(1, 12, 12)
#' result <- compute_saliency_features(img, sal, "weights", pairwise = FALSE,
#'   filter_length = 5L, gabor_bins = 4L, transform_controls = FALSE)
#' result$features[, c("control", "first_order", "effective_fraction")]
#' @export
compute_saliency_features <- function(image, saliency, representation,
    signed_policy = "reject", image_id = if (is.character(image)) image else "image",
    max_pixels = 120000L, gabor_bins = 24L, filter_length = 31L,
    maxdiag = 500L, circ_bins = 48L,
    ranges = list(c(20L, 80L), c(80L, 160L), c(160L, 240L)),
    pairwise = TRUE, pair_weighting = "both", edge_selection = "image",
    validity = "legacy", min_cell_mass_fraction = NULL, min_effective_pairs = NULL,
    centre_sigma = .25, transform_controls = TRUE,
    donor_maps = list(), donor_assignments = NULL, scales = NULL,
    scale_units = "relative", pixels_per_degree = NULL, provenance = list(), use_cpp = TRUE,
    centre_match = FALSE, donor_aspect_tolerance = .01, min_common_coverage = 0) {
  if (missing(representation)) stop("Specify representation explicitly")
  representation <- match.arg(representation, c("weights", "log_density"))
  signed_policy <- match.arg(signed_policy, c("reject", "rectify", "shift"))
  pair_weighting <- match.arg(pair_weighting, c("both", "reference"))
  edge_selection <- match.arg(edge_selection, c("image", "saliency"))
  validity <- match.arg(validity, c("legacy", "support"))
  checkmate::assert_string(image_id, min.chars = 1L)
  checkmate::assert_flag(pairwise); checkmate::assert_flag(transform_controls)
  checkmate::assert_flag(use_cpp); checkmate::assert_list(provenance)
  checkmate::assert_flag(centre_match)
  checkmate::assert_number(donor_aspect_tolerance, lower = 0, upper = 1, finite = TRUE)
  checkmate::assert_number(min_common_coverage, lower = 0, upper = 1, finite = TRUE)
  gabor_bins <- .saliency_positive_integer(gabor_bins, "gabor_bins")
  filter_length <- .saliency_positive_integer(filter_length, "filter_length")
  if (filter_length %% 2L == 0L) stop("filter_length must be odd")
  maxdiag <- .saliency_positive_integer(maxdiag, "maxdiag")
  circ_bins <- .saliency_positive_integer(circ_bins, "circ_bins")
  if (as.double(maxdiag) * circ_bins * gabor_bins > .Machine$integer.max) stop("Histogram dimensions too large")
  if (!is.null(max_pixels)) max_pixels <- .saliency_positive_integer(max_pixels, "max_pixels")
  if (!is.list(ranges) || !length(ranges)) stop("ranges must be a nonempty list")
  ranges <- lapply(ranges, function(r) {
    if (!is.numeric(r) || length(r) != 2 || any(!is.finite(r)) ||
        any(r != floor(r)) || r[1] < 1 || r[2] < r[1] || any(r > .Machine$integer.max))
      stop("Each range must be two positive, increasing integer indices")
    as.integer(r)
  })
  if (validity == "support") {
    checkmate::assert_number(min_cell_mass_fraction, lower = 0, upper = 1)
    checkmate::assert_number(min_effective_pairs, lower = 0, finite = TRUE)
  }
  map <- .saliency_map(saliency, representation, signed_policy)
  if (is.character(image) && length(image) == 1L) {
    img <- imager::load.image(image)
    original_shape <- dim(img)[1:2]
    if (!identical(as.integer(dim(saliency)), as.integer(original_shape)))
      stop("saliency dimensions must match the original image")
    if (!is.null(max_pixels)) {
      shape <- as.integer(original_shape * sqrt(max_pixels / prod(original_shape)))
      if (any(shape < 1L)) stop("Requested resize would produce an empty image")
      img <- imager::resize(img, shape[1], shape[2], interpolation_type = 6)
    }
    shape <- dim(img)[1:2]
    grey <- if (dim(img)[4] == 1L) img else imager::grayscale(img)
    image_matrix <- matrix(as.array(grey), shape[1], shape[2])
  } else {
    .saliency_matrix(image, "image")
    image_matrix <- image
    original_shape <- shape <- dim(image)
    if (!identical(dim(saliency), shape)) stop("saliency dimensions must match the image")
  }
  if (any(shape < 3L) || any(!is.finite(image_matrix))) stop("Image must be finite and at least 3 by 3")
  location_before <- .saliency_location(map$q, map$status)
  map$q <- .saliency_resize(map$q, shape)
  if (map$status == "ok" && !any(map$q > 0)) map$status <- "zero_map_mass"
  controls <- .saliency_controls(map$q, map$status, centre_sigma, transform_controls,
    donor_maps, donor_assignments, image_id, representation, signed_policy, original_shape,
    centre_match, donor_aspect_tolerance)
  controls$observed$metadata <- list(source_dimensions = original_shape, analysis_dimensions = shape,
    map_interpolation = .saliency_resize_method(original_shape, shape),
    location_before_resize = location_before, location_after_resize = .saliency_location(map$q, map$status))
  if (use_cpp) {
    filtered <- saliency_filter_cpp(image_matrix, gabor_bins, filter_length)
    bins <- filtered$bins_vec
  } else {
    bank <- filter_bank(gabor_bins, filter_length)
    filtered <- run_filterbank(list(image_raw = image_matrix, image_size = shape), bank)
    bins <- bank$bins_vec
  }
  a <- filtered$resp_val
  orientation <- filtered$resp_bin
  if (any(!is.finite(a) | a < 0)) stop("Edge responses must be finite and nonnegative; use an orientation bank covering the full circle")
  first_set <- a > 0
  fixed_pairs <- .saliency_selection(a, map$q, "image")
  features <- pair_ranges <- cells <- multiscale <- selected <- list()
  evaluation_order <- c("unweighted", setdiff(names(controls), "unweighted"))
  prepared <- groups <- list()
  for (id in evaluation_order) {
    control <- controls[[id]]
    q <- if (control$status == "ok") control$q else matrix(0, shape[1], shape[2])
    pair_set <- if (edge_selection == "image" || control$status != "ok") fixed_pairs else
      .saliency_selection(a, q, "saliency")
    t1 <- .saliency_edge_weights(q, first_set)
    t2 <- .saliency_edge_weights(q, pair_set)
    prepared[[id]] <- list(selected = pair_set, t1 = t1, t2 = t2)
    group <- which(vapply(groups, function(ids) identical(pair_set, prepared[[ids[1]]]$selected), logical(1)))
    if (length(group)) groups[[group[1]]] <- c(groups[[group[1]]], id) else groups[[length(groups) + 1L]] <- id
  }
  # One geometry pass and one baseline per distinct edge set. The default
  # image-selected mode has exactly one group, including all donor controls.
  counted_maps <- baselines <- list()
  for (ids in groups) {
    first_weights <- lapply(prepared[ids], function(x) x$t1$weights)
    pair_weights <- lapply(prepared[ids], function(x) x$t2$weights)
    counter <- if (use_cpp) saliency_count_batch_cpp else .saliency_count_batch_r
    batch <- suppressWarnings(counter(orientation, a, first_weights, pair_weights,
      prepared[[ids[1]]]$selected, gabor_bins, maxdiag, circ_bins, pairwise, pair_weighting == "reference"))
    names(batch) <- ids
    counted_maps[ids] <- batch
    if (pairwise) {
      base <- .saliency_cell_stats(batch[[1]]$base_counts, batch[[1]]$base_squares,
        batch[[1]]$base_pairs, bins, use_cpp, validity, min_cell_mass_fraction, min_effective_pairs)
      baselines[ids] <- rep(list(base), length(ids))
    }
  }
  for (id in evaluation_order) {
    control <- controls[[id]]
    q <- control$q
    pair_set <- prepared[[id]]$selected
    t1 <- prepared[[id]]$t1
    t2 <- prepared[[id]]$t2
    first_status <- if (control$status != "ok") control$status else t1$status
    pair_status <- if (control$status != "ok") control$status else t2$status
    counted <- counted_maps[[id]]
    loc <- .saliency_location(q, control$status)
    features[[id]] <- data.frame(image_id = image_id, control = id, family = control$family,
      donor_id = control$donor_id, replicate = control$replicate,
      first_order = if (first_status == "ok") counted$first_order else NA_real_,
      first_order_status = first_status, H_location = loc$H_location,
      KL_uniform = loc$KL_uniform, effective_fraction = loc$effective_fraction,
      location_status = loc$status, complex_before = sum(a) / length(a),
      edge_mass_first = if (control$status == "ok") sum(q[first_set]) else NA_real_,
      edge_mass_pair = if (control$status == "ok") sum(q[pair_set]) else NA_real_,
      pair_status = if (pairwise) pair_status else "not_requested",
      identity_equivalent = control$identity_equivalent, duplicate_of = control$duplicate_of,
      stringsAsFactors = FALSE)
    selected[[id]] <- which(pair_set, arr.ind = TRUE)
    if (pairwise) {
      summary <- .saliency_pair_summary(counted, bins, ranges, use_cpp, validity,
                                         min_cell_mass_fraction, min_effective_pairs, baselines[[id]])
      if (pair_status != "ok") {
        summary$ranges$entropy <- NA_real_
        summary$ranges$status <- pair_status
        summary$cells$status <- pair_status
        summary$cells$valid <- FALSE
      }
      pair_ranges[[id]] <- cbind(image_id = image_id, control = id, family = control$family, summary$ranges)
      cells[[id]] <- summary$cells
    }
    if (!is.null(scales)) multiscale[[id]] <- .saliency_multiscale(q, control$status,
      scales, scale_units, pixels_per_degree, rbind(c(0,0), c(.5,0), c(0,.5), c(.5,.5)))
  }
  feature_table <- do.call(rbind, features[names(controls)])
  rownames(feature_table) <- NULL
  feature_table$observed_minus_control <- feature_table$first_order[1] - feature_table$first_order
  range_table <- if (pairwise) do.call(rbind, pair_ranges[names(controls)]) else data.frame()
  if (pairwise) {
    rownames(range_table) <- NULL
    observed <- pair_ranges$observed$entropy
    range_table$observed_minus_control <- observed[range_table$range_id] - range_table$entropy
    range_table <- .saliency_common_contrasts(range_table, cells, ranges, min_common_coverage)
  }
  comparisons <- .saliency_family_comparisons(feature_table, range_table, cells, controls,
                                               ranges, min_common_coverage)
  for (family in c("transformed", "donor")) {
    column <- paste0("observed_minus_", if (family == "transformed") "transform" else "donor", "_mean")
    feature_table[[column]] <- NA_real_
    cmp <- comparisons[comparisons$family == family, ]
    feature_table[[column]][1] <- cmp$observed_minus_control_mean[cmp$metric == "first_order"]
    if (pairwise) {
      range_table[[column]] <- NA_real_
      range_table[[column]][range_table$control == "observed"] <-
        cmp$observed_minus_control_mean[cmp$metric == "pairwise"]
    }
  }
  list(features = feature_table, ranges = range_table, comparisons = comparisons,
    cells = if (pairwise) cells[names(controls)] else list(),
    multiscale = multiscale[names(multiscale)], selection = selected[names(controls)],
    metadata = list(representation = representation, signed_policy = signed_policy,
      edge_selection = edge_selection, pair_weighting = pair_weighting, validity = validity,
      min_cell_mass_fraction = min_cell_mass_fraction, min_effective_pairs = min_effective_pairs,
      backend = if (use_cpp) "cpp" else "R", geometry = "euclidean",
      min_common_coverage = min_common_coverage, centre_match = centre_match,
      donor_aspect_tolerance = donor_aspect_tolerance, counting_groups = length(groups),
      original_dimensions = original_shape, analysis_dimensions = shape,
      map_interpolation = .saliency_resize_method(original_shape, shape),
      filter = list(gabor_bins = gabor_bins, filter_length = filter_length),
      pair_bins = list(maxdiag = maxdiag, circ_bins = circ_bins, ranges = ranges),
      added_smoothing = "none", provenance = provenance,
      controls = lapply(controls, function(x) x$metadata)))
}
