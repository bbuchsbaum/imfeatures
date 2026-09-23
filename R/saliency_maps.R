# Shared validation deliberately does not infer map representation from its range.
.saliency_matrix <- function(x, name = "saliency") {
  if (!is.matrix(x) || !is.numeric(x) || any(dim(x) < 1L)) {
    stop(name, " must be a nonempty numeric matrix", call. = FALSE)
  }
}

.saliency_map <- function(x, representation, signed_policy = "reject") {
  .saliency_matrix(x)
  representation <- match.arg(representation, c("weights", "log_density"))
  signed_policy <- match.arg(signed_policy, c("reject", "rectify", "shift"))
  if (representation == "log_density") {
    if (signed_policy != "reject") stop("Signed policies apply only to raw weights")
    if (anyNA(x) || any(x == Inf)) stop("Log maps must contain finite values or -Inf")
    if (all(x == -Inf)) return(list(q = matrix(0, nrow(x), ncol(x)), status = "zero_map_mass"))
    x <- exp(x - max(x))
  } else {
    if (any(!is.finite(x))) stop("Raw map weights must be finite")
    # Scale before shifting so finite extreme signed inputs cannot overflow.
    scale <- max(abs(x))
    if (scale > 0) x <- x / scale
    if (any(x < 0)) {
      if (signed_policy == "reject") stop("Negative raw weights require an explicit signed_policy")
      if (signed_policy == "rectify") x <- pmax(x, 0)
      if (signed_policy == "shift") x <- x - min(x)
    }
  }
  if (!any(x > 0)) return(list(q = matrix(0, nrow(x), ncol(x)), status = "zero_map_mass"))
  x <- x / max(x)
  list(q = x / sum(x), status = "ok")
}

.saliency_location <- function(q, status = "ok") {
  if (status != "ok") return(data.frame(H_location = NA_real_, KL_uniform = NA_real_,
                                        effective_fraction = NA_real_, status = status))
  p <- q[q > 0]
  h <- -sum(p * log2(p))
  kl <- max(0, log2(length(q)) - h)
  data.frame(H_location = h, KL_uniform = kl, effective_fraction = 2^-kl, status = status)
}

.saliency_resize <- function(q, shape) {
  if (identical(as.integer(dim(q)), as.integer(shape))) return(q)
  # Average before subsampling; linear interpolation can miss a narrow hotspot.
  # Mixed-axis resizes shrink first, then interpolate only the growing axes.
  smaller <- pmin(dim(q), shape)
  out <- imager::as.cimg(q)
  if (any(smaller < dim(q)))
    out <- imager::resize(out, smaller[1], smaller[2], interpolation_type = 2)
  if (any(shape > smaller))
    out <- imager::resize(out, shape[1], shape[2], interpolation_type = 3)
  out <- as.array(out)
  out <- matrix(out, shape[1], shape[2])
  if (sum(out) > 0) out <- out / sum(out)
  out
}

.saliency_resize_method <- function(from, to) {
  if (all(from == to)) "none" else if (all(to <= from)) "area" else
    if (all(to >= from)) "linear" else "area_then_linear"
}

.saliency_positive_integer <- function(x, name) {
  if (!is.numeric(x) || length(x) != 1L || !is.finite(x) || x < 1 ||
      x != floor(x) || x > .Machine$integer.max) stop(name, " must be a positive integer")
  as.integer(x)
}

#' Saliency location entropy and effective attended fraction
#'
#' Normalizes a heat map over all pixels, separately from edge weighting.
#' A zero-mass map has unavailable features. Negative log densities are valid.
#' @md
#' @param saliency Nonempty numeric matrix of aligned saliency values.
#' @param representation Required: `"weights"` or `"log_density"` (natural logs).
#' @param signed_policy For raw weights: `"reject"`, `"rectify"`, or `"shift"`.
#' @return A data frame containing `H_location` and `KL_uniform` in bits,
#'   `effective_fraction = 2^-KL_uniform`, and `status`.
#' @details KL and effective fraction are invariant to equal-mass subdivision
#'   of every pixel. They are not invariant to arbitrary resampling or smoothing.
#' @examples
#' saliency_location_entropy(matrix(1, 8, 8), "weights")
#' @export
saliency_location_entropy <- function(saliency, representation, signed_policy = "reject") {
  if (missing(representation)) stop("Specify representation explicitly")
  m <- .saliency_map(saliency, representation, signed_policy)
  .saliency_location(m$q, m$status)
}

#' Multiscale location entropy of a saliency map
#'
#' Sums probability mass into clipped spatial blocks at several scales and
#' shifted origins. This describes one location distribution, not joint entropy.
#' @md
#' @inheritParams saliency_location_entropy
#' @param scales Positive block sizes, in the units selected by `units`.
#' @param units `"relative"` (fraction of the shorter image dimension),
#'   `"degrees"`, or explicit `"pixels"`.
#' @param pixels_per_degree Positive conversion factor required for degrees.
#' @param origins Two-column matrix of offsets as fractions of one block, in
#'   matrix-axis order. Values must be in `[0,1)`.
#' @return A list with origin-specific `curve`, origin-averaged `summary`, and
#'   `metadata`. Boundary blocks use their actual areas for KL from uniform.
#' @examples
#' saliency_multiscale_entropy(matrix(1, 8, 8), "weights", scales = c(.25, .5))
#' @export
saliency_multiscale_entropy <- function(saliency, representation, signed_policy = "reject",
                                      scales = c(1/32, 1/16, 1/8, 1/4, 1/2),
                                      units = "relative", pixels_per_degree = NULL,
                                      origins = rbind(c(0, 0), c(.5, 0), c(0, .5), c(.5, .5))) {
  if (missing(representation)) stop("Specify representation explicitly")
  m <- .saliency_map(saliency, representation, signed_policy)
  .saliency_multiscale(m$q, m$status, scales, units, pixels_per_degree, origins)
}

.saliency_multiscale <- function(q, status, scales, units, pixels_per_degree, origins) {
  units <- match.arg(units, c("relative", "degrees", "pixels"))
  if (!is.numeric(scales) || !length(scales) || any(!is.finite(scales) | scales <= 0))
    stop("scales must be finite and positive")
  if (!is.matrix(origins) || !is.numeric(origins) || ncol(origins) != 2L ||
      !nrow(origins) || any(!is.finite(origins) | origins < 0 | origins >= 1))
    stop("origins must have two columns with values in [0,1)")
  if (units == "degrees" && (length(pixels_per_degree) != 1L ||
      !is.numeric(pixels_per_degree) || !is.finite(pixels_per_degree) || pixels_per_degree <= 0))
    stop("degrees require a positive pixels_per_degree")
  multiplier <- switch(units, relative = min(dim(q)), degrees = pixels_per_degree, pixels = 1)
  pixels <- pmax(1, round(scales * multiplier))
  if (any(!is.finite(pixels) | pixels > .Machine$integer.max)) stop("Block sizes are too large")
  rows <- vector("list", length(scales) * nrow(origins))
  z <- 0L
  for (b in seq_along(scales)) for (o in seq_len(nrow(origins))) {
    block <- pixels[b]
    shift <- floor(origins[o, ] * block)
    gx <- floor((seq_len(nrow(q)) - 1 - shift[1]) / block)
    gy <- floor((seq_len(ncol(q)) - 1 - shift[2]) / block)
    group <- as.integer(factor(as.vector(outer(gx, gy, function(x, y) paste(x, y, sep = ":")))))
    mass <- as.numeric(rowsum(as.vector(q), group, reorder = TRUE))
    area <- tabulate(group) / length(q)
    positive <- mass > 0
    h <- if (status == "ok") -sum(mass[positive] * log2(mass[positive])) else NA_real_
    kl <- if (status == "ok") max(0, sum(mass[positive] * log2(mass[positive] / area[positive]))) else NA_real_
    z <- z + 1L
    rows[[z]] <- data.frame(scale_index = b, scale = scales[b], block_pixels = block,
      origin = o, offset_1 = shift[1], offset_2 = shift[2], blocks = length(mass),
      H_location = h, KL_uniform = kl, effective_fraction = 2^-kl, status = status)
  }
  curve <- do.call(rbind, rows)
  summary <- do.call(rbind, lapply(seq_along(scales), function(b) {
    v <- curve[curve$scale_index == b, ]
    data.frame(scale_index = b, scale = scales[b], block_pixels = pixels[b],
      H_location = mean(v$H_location), KL_uniform = mean(v$KL_uniform),
      effective_fraction = mean(v$effective_fraction), status = status)
  }))
  list(curve = curve, summary = summary,
       metadata = list(units = units, pixels_per_degree = pixels_per_degree,
                       origins = origins, dimensions = dim(q), boundary = "clipped"))
}
