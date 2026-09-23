.saliency_selection <- function(a, q, selection) {
  score <- a
  if (selection == "saliency") score <- a * if (max(q) > 0) q / max(q) else q
  rank <- if (length(score) >= 10000L) 10000L else max(1L, ceiling(length(score) * .1))
  cutoff <- sort(as.vector(score), decreasing = TRUE)[rank]
  a > 0 & score >= cutoff
}

.saliency_edge_weights <- function(q, selected) {
  out <- matrix(0, nrow(q), ncol(q))
  if (!any(selected)) return(list(weights = out, status = "no_edges"))
  values <- q[selected]
  if (!any(values > 0)) return(list(weights = out, status = "zero_edge_saliency"))
  if (all(values == values[1])) {
    out[selected] <- 1
  } else {
    values <- values / max(values)
    out[selected] <- values / mean(values)
  }
  list(weights = out, status = "ok")
}

.saliency_count_r <- function(orientation, response, first_weights, pair_weights,
                              selected, num_filters, maxdiag, circ_bins,
                              pairwise, reference_only, r_geometry = FALSE) {
  weighted <- response * first_weights
  if (any(!is.finite(weighted)) || !is.finite(sum(weighted))) stop("Weighted response overflow")
  first <- first_order_entropy(list(resp_val = weighted, resp_bin = orientation,
                                   num_filters = num_filters))
  if (!pairwise) return(list(first_order = first))
  counts <- base_counts <- array(0, c(maxdiag, circ_bins, num_filters))
  squares <- base_squares <- pairs <- base_pairs <- matrix(0, maxdiag, circ_bins)
  coords <- which(selected & response > 0, arr.ind = TRUE)
  if (!nrow(coords)) return(list(first_order = first, counts = counts, squares = squares,
    pairs = pairs, base_counts = base_counts, base_squares = base_squares, base_pairs = base_pairs))
  a <- response[coords]
  t <- pair_weights[coords]
  orientations <- orientation[coords]
  add <- function(target, values, index) {
    accum <- rowsum(values, index, reorder = FALSE)
    ids <- as.integer(rownames(accum))
    target[ids] <- target[ids] + accum[, 1]
    target
  }
  for (cp in seq_len(nrow(coords))) {
    dx <- coords[, 1] - coords[cp, 1]
    dy <- coords[, 2] - coords[cp, 2]
    offset <- as.integer(r_geometry)
    distance <- pmin(maxdiag, round(sqrt((dx - offset)^2 + (dy - offset)^2)) + 1)
    direction <- round(atan2(dy, dx) / (2*pi) * circ_bins + orientations[cp] / num_filters * circ_bins) %% circ_bins
    relative <- (orientations - orientations[cp] + num_filters) %% num_filters
    cell <- distance + maxdiag * direction
    index <- cell + maxdiag * circ_bins * relative
    base <- a * a[cp]
    weights <- base * t[cp]
    if (!reference_only) weights <- weights * t
    if (any(!is.finite(base^2)) || any(!is.finite(weights^2))) stop("Pair weight overflow")
    counts <- add(counts, weights, index)
    base_counts <- add(base_counts, base, index)
    squares <- add(squares, weights^2, cell)
    base_squares <- add(base_squares, base^2, cell)
    pairs <- add(pairs, as.numeric(weights > 0), cell)
    base_pairs <- add(base_pairs, as.numeric(base > 0), cell)
  }
  list(first_order = first, counts = counts, squares = squares, pairs = pairs,
       base_counts = base_counts, base_squares = base_squares, base_pairs = base_pairs)
}

.saliency_ratio <- function(numerator, denominator) {
  ifelse(denominator > 0, numerator / denominator, NA_real_)
}

# Slow reference backend: reuse geometry and the baseline across controls too.
.saliency_count_batch_r <- function(orientation, response, first_weights, pair_weights,
                                     selected, num_filters, maxdiag, circ_bins,
                                     pairwise, reference_only) {
  out <- lapply(seq_along(first_weights), function(k) .saliency_count_r(orientation,
    response, first_weights[[k]], pair_weights[[k]], selected, num_filters,
    maxdiag, circ_bins, FALSE, reference_only))
  if (!pairwise) return(out)
  shape <- c(maxdiag, circ_bins, num_filters)
  base_counts <- array(0, shape)
  base_squares <- base_pairs <- matrix(0, maxdiag, circ_bins)
  for (k in seq_along(out)) {
    out[[k]]$counts <- array(0, shape)
    out[[k]]$squares <- matrix(0, maxdiag, circ_bins)
    out[[k]]$pairs <- matrix(0, maxdiag, circ_bins)
  }
  coords <- which(selected & response > 0, arr.ind = TRUE)
  a <- response[coords]; orientations <- orientation[coords]
  t <- lapply(pair_weights, function(w) w[coords])
  add <- function(target, values, index) {
    accum <- rowsum(values, index, reorder = FALSE)
    ids <- as.integer(rownames(accum))
    target[ids] <- target[ids] + accum[, 1]
    target
  }
  for (cp in seq_len(nrow(coords))) {
    dx <- coords[, 1] - coords[cp, 1]; dy <- coords[, 2] - coords[cp, 2]
    distance <- pmin(maxdiag, round(sqrt(dx^2 + dy^2)) + 1)
    direction <- round(atan2(dy, dx) / (2*pi) * circ_bins + orientations[cp] / num_filters * circ_bins) %% circ_bins
    relative <- (orientations - orientations[cp] + num_filters) %% num_filters
    cell <- distance + maxdiag * direction
    index <- cell + maxdiag * circ_bins * relative
    base <- a * a[cp]
    if (any(!is.finite(base^2))) stop("Pair weight overflow")
    base_counts <- add(base_counts, base, index)
    base_squares <- add(base_squares, base^2, cell)
    base_pairs <- add(base_pairs, as.numeric(base > 0), cell)
    for (k in seq_along(out)) {
      weights <- base * t[[k]][cp]
      if (!reference_only) weights <- weights * t[[k]]
      if (any(!is.finite(weights^2))) stop("Pair weight overflow")
      out[[k]]$counts <- add(out[[k]]$counts, weights, index)
      out[[k]]$squares <- add(out[[k]]$squares, weights^2, cell)
      out[[k]]$pairs <- add(out[[k]]$pairs, as.numeric(weights > 0), cell)
    }
  }
  lapply(out, function(x) c(x, list(base_counts = base_counts, base_squares = base_squares,
                                  base_pairs = base_pairs)))
}

.saliency_cell_stats <- function(counts, squares, pairs, bins, use_cpp,
                                 validity, min_mass, min_pairs) {
  shape <- dim(counts)[1:2]
  m <- matrix(counts, prod(shape), length(bins))
  mass <- rowSums(m)
  sq <- as.vector(squares)
  if (any(!is.finite(mass)) || any(!is.finite(sq))) stop("Pair accumulation overflow")
  neff <- ifelse(sq > 0, (mass / sqrt(sq))^2, NA_real_)
  fraction <- .saliency_ratio(mass, sum(mass))
  if (validity == "legacy") {
    if (use_cpp) {
      stats <- saliency_statistics_cpp(counts, bins)
      h <- as.vector(stats$shannon)
      valid <- !is.na(as.vector(stats$shannon_nan)) & mass > 0
    } else {
      # Match the native legacy rule, including its floating-point offset
      # arithmetic, without inheriting the historical R validity difference.
      h <- apply(m, 1L, function(x) {
        if (!sum(x)) return(NA_real_)
        p <- x / (sum(x) + 1e-5)
        p <- p / sum(p)
        -sum(p[p > 0] * log2(p[p > 0]))
      })
      valid <- (mass + 1e-5) - 1e-5 > 1 & mass > 0
    }
  } else {
    p <- m / ifelse(mass > 0, mass, 1)
    h <- -rowSums(ifelse(p > 0, p * log2(p), 0))
    h[mass == 0] <- NA_real_
    valid <- mass > 0 & !is.na(neff) & !is.na(fraction) &
      fraction >= min_mass & neff >= min_pairs
  }
  data.frame(mass = mass, squared_mass = sq, positive_pairs = as.vector(pairs),
    occupied_orientations = rowSums(m > 0), mass_fraction = fraction,
    effective_pairs = neff, entropy = h, valid = valid,
    status = ifelse(mass == 0, "zero_cell_mass", ifelse(valid, "ok", "insufficient_support")))
}

.saliency_pair_summary <- function(counted, bins, ranges, use_cpp, validity, min_mass, min_pairs,
                                    baseline = NULL) {
  shape <- dim(counted$counts)[1:2]
  cell <- .saliency_cell_stats(counted$counts, counted$squares, counted$pairs,
                              bins, use_cpp, validity, min_mass, min_pairs)
  base <- if (!is.null(baseline)) baseline else .saliency_cell_stats(counted$base_counts, counted$base_squares, counted$base_pairs,
                              bins, use_cpp, validity, min_mass, min_pairs)
  means <- function(v) {
    h <- matrix(ifelse(v$valid, v$entropy, NA_real_), shape[1], shape[2])
    if (use_cpp) return(saliency_range_means_cpp(h, ranges))
    rows <- rowMeans(h, na.rm = TRUE)
    vapply(ranges, function(r) {
      value <- mean(rows[intersect(seq.int(r[1], r[2]), seq_len(shape[1]))], na.rm = TRUE)
      if (is.finite(value)) value else NA_real_
    }, numeric(1))
  }
  h <- means(cell)
  bh <- means(base)
  dimensions <- expand.grid(distance_index = seq_len(shape[1]), direction_bin = seq_len(shape[2]) - 1L)
  range_metrics <- function(v, ids) {
    m <- sum(v$mass[ids]); q <- sum(v$squared_mass[ids])
    c(mass = m, squared_mass = q, positive_pairs = sum(v$positive_pairs[ids]),
      mass_fraction = .saliency_ratio(m, sum(v$mass)),
      effective_pairs = if (q > 0) (m / sqrt(q))^2 else NA_real_,
      valid_cell_coverage = if (length(ids)) mean(v$valid[ids]) else NA_real_,
      retained_mass_fraction = .saliency_ratio(sum(v$mass[ids][v$valid[ids]]), m))
  }
  out <- lapply(seq_along(ranges), function(i) {
    r <- ranges[[i]]
    ids <- which(dimensions$distance_index >= r[1] & dimensions$distance_index <= r[2])
    w <- range_metrics(cell, ids); b <- range_metrics(base, ids)
    data.frame(range_id = i, start = r[1], end = r[2], entropy = h[i],
      unweighted_entropy = bh[i], status = if (is.na(h[i])) "insufficient_support" else "ok",
      as.list(w), stats::setNames(as.list(b), paste0("unweighted_", names(b))),
      effective_pairs_ratio = .saliency_ratio(w["effective_pairs"], b["effective_pairs"]),
      row.names = NULL, check.names = FALSE)
  })
  names(base) <- paste0("unweighted_", names(base))
  list(ranges = do.call(rbind, out), cells = cbind(dimensions, cell, base,
       effective_pairs_ratio = .saliency_ratio(cell$effective_pairs, base$unweighted_effective_pairs)))
}
