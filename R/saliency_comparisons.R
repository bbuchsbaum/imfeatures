# Intersect validity before reducing: equal directions within distance, then
# equal distances, just as in the descriptive range summaries. Every compared
# map therefore uses exactly the same cells and aggregation weights.
.saliency_common_summary <- function(cells, range, min_coverage) {
  first <- cells[[1]]
  within <- first$distance_index >= range[1] & first$distance_index <= range[2]
  common <- within & Reduce(`&`, lapply(cells, function(x) x$valid))
  coverage <- if (any(within)) mean(common[within]) else NA_real_
  distances <- unique(first$distance_index[within])
  distance_coverage <- if (length(distances)) length(unique(first$distance_index[common])) / length(distances) else NA_real_
  ok <- any(common) && !is.na(coverage) && coverage >= min_coverage
  values <- vapply(cells, function(x) {
    if (!ok) return(NA_real_)
    mean(tapply(x$entropy[common], x$distance_index[common], mean))
  }, numeric(1))
  retained <- vapply(cells, function(x) .saliency_ratio(sum(x$mass[common]), sum(x$mass[within])), numeric(1))
  list(values = values, coverage = coverage, distance_coverage = distance_coverage,
       retained = retained, status = if (ok) "ok" else "insufficient_common_support")
}

.saliency_common_contrasts <- function(table, cells, ranges, min_coverage) {
  common <- lapply(seq_len(nrow(table)), function(i) {
    id <- table$control[i]
    x <- .saliency_common_summary(cells[c("observed", id)], ranges[[table$range_id[i]]], min_coverage)
    data.frame(common_observed_entropy = x$values[1], common_control_entropy = x$values[2],
      observed_minus_control_common = unname(x$values[1] - x$values[2]),
      common_valid_cell_coverage = x$coverage, common_distance_coverage = x$distance_coverage,
      common_observed_retained_mass = x$retained[1], common_control_retained_mass = x$retained[2],
      common_status = x$status, row.names = NULL)
  })
  cbind(table, do.call(rbind, common))
}

.saliency_family_comparisons <- function(features, range_table, cells, controls, ranges, min_coverage) {
  rows <- list()
  for (family in c("transformed", "donor")) {
    candidates <- features$control[features$family == family]
    metrics <- c("first_order", if (nrow(range_table)) rep("pairwise", length(ranges)))
    for (j in seq_along(metrics)) {
      metric <- metrics[j]
      range_id <- if (metric == "pairwise") j - 1L else NA_integer_
      eligible <- if (metric == "first_order") {
        features$control[features$first_order_status == "ok"]
      } else range_table$control[range_table$range_id == range_id & range_table$status == "ok"]
      available <- intersect(candidates, eligible)
      # Deduplicate within each family, not across families. Repeated donor
      # assignments and symmetric transforms are not independent null draws.
      used <- character()
      for (id in available) {
        if (controls[[id]]$identity_equivalent) next
        duplicate <- any(vapply(used, function(other) all(controls[[id]]$q == controls[[other]]$q), logical(1)))
        if (!duplicate) used <- c(used, id)
      }
      observed <- control_mean <- difference <- control_sd <- control_min <- control_max <- NA_real_
      coverage <- distance_coverage <- observed_retained <- control_retained <- NA_real_
      status <- if (!"observed" %in% eligible) "unavailable_observed" else
        if (!length(used)) "no_distinct_controls" else "ok"
      if (status == "ok") {
        if (metric == "first_order") {
          values <- features$first_order[match(c("observed", used), features$control)]
        } else {
          x <- .saliency_common_summary(cells[c("observed", used)], ranges[[range_id]], min_coverage)
          values <- x$values
          coverage <- x$coverage; distance_coverage <- x$distance_coverage
          observed_retained <- x$retained[1]; control_retained <- min(x$retained[-1])
          status <- x$status
        }
        if (status == "ok") {
          observed <- values[1]
          control_mean <- mean(values[-1])
          difference <- observed - control_mean
          control_sd <- stats::sd(values[-1])
          control_min <- min(values[-1]); control_max <- max(values[-1])
        }
      }
      rows[[length(rows) + 1L]] <- data.frame(image_id = features$image_id[1], family = family,
        metric = metric, range_id = range_id,
        start = if (is.na(range_id)) NA_integer_ else ranges[[range_id]][1],
        end = if (is.na(range_id)) NA_integer_ else ranges[[range_id]][2],
        observed = observed, control_mean = control_mean, observed_minus_control_mean = difference,
        control_sd = control_sd, control_min = control_min, control_max = control_max,
        n_requested = length(candidates), n_available = length(available), n_controls = length(used),
        controls_used = paste(used, collapse = ", "), common_valid_cell_coverage = coverage,
        common_distance_coverage = distance_coverage, common_observed_retained_mass = observed_retained,
        minimum_control_retained_mass = control_retained, status = status, row.names = NULL)
    }
  }
  do.call(rbind, rows)
}
