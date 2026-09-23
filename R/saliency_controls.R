#' Reproducible donor-map permutations for saliency controls
#'
#' Produces one-to-one assignments within declared strata, excluding the same
#' stimulus (including alternate versions). A stratum should encode aspect
#' ratio, saliency model/version, and preprocessing. No pixel shuffling is used.
#' These strata do not automatically match concentration; include a declared
#' concentration stratum if the study requires it, and inspect achieved matches
#' after resizing.
#' @md
#' @param ids Unique image/map identifiers.
#' @param strata Stratum identifier for each image.
#' @param stimulus_ids Underlying stimulus identifiers; shared values exclude
#'   alternate versions of the same stimulus from being each other's donors.
#' @param replicates Number of randomized assignments per stratum.
#' @param seed Local random seed; the caller's random state is restored.
#' @return A data frame with target/donor IDs, stimulus IDs, replicate, stratum,
#'   seed, eligible donor count, and status. A stratum without a complete valid
#'   assignment returns unavailable rows, not an invalid partial permutation.
#' @details Randomized matching does not sample all valid permutations uniformly.
#'   These assignments support descriptive controls, not calibrated permutation
#'   p-values. Repeated assignments are possible, especially in small strata.
#' @examples
#' saliency_donor_assignments(c("a", "b", "c"), rep("same-model-square", 3), seed = 3)
#' @export
saliency_donor_assignments <- function(ids, strata, stimulus_ids = ids,
                                       replicates = 10L, seed = 1L) {
  valid_ids <- function(x) is.character(x) && length(x) == length(ids) && !anyNA(x) && all(nzchar(x))
  if (!length(ids) || !valid_ids(ids) || anyDuplicated(ids)) stop("ids must be unique nonempty strings")
  if (!valid_ids(strata) || !valid_ids(stimulus_ids)) stop("strata and stimulus_ids must match ids")
  replicates <- .saliency_positive_integer(replicates, "replicates")
  if (!is.numeric(seed) || length(seed) != 1 || !is.finite(seed) || seed != floor(seed) ||
      abs(seed) > .Machine$integer.max) stop("seed must be a finite integer")
  existed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (existed) previous <- get(".Random.seed", envir = .GlobalEnv)
  on.exit(if (existed) assign(".Random.seed", previous, envir = .GlobalEnv) else
    if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) rm(".Random.seed", envir = .GlobalEnv))
  set.seed(seed)
  shuffle <- function(v) if (length(v)) v[sample.int(length(v))] else v
  rows <- list()
  for (stratum in unique(strata)) {
    group <- which(strata == stratum)
    candidates <- lapply(group, function(i) which(stimulus_ids[group] != stimulus_ids[i]))
    for (replicate in seq_len(replicates)) {
      adjacency <- lapply(candidates, shuffle)
      owner <- rep(NA_integer_, length(group))
      # Randomized augmenting paths find a complete constrained matching when
      # one exists, without unbounded rejection sampling of permutations.
      visit <- function(target) {
        for (donor in adjacency[[target]]) {
          if (seen[donor]) next
          seen[donor] <<- TRUE
          if (is.na(owner[donor]) || visit(owner[donor])) {
            owner[donor] <<- target
            return(TRUE)
          }
        }
        FALSE
      }
      complete <- TRUE
      for (target in shuffle(seq_along(group))) {
        seen <- rep(FALSE, length(group))
        if (!visit(target)) { complete <- FALSE; break }
      }
      assignment <- rep(NA_integer_, length(group))
      if (complete) assignment[owner] <- seq_along(group)
      rows[[length(rows) + 1L]] <- data.frame(target_id = ids[group],
        target_stimulus_id = stimulus_ids[group], donor_id = ids[group[assignment]],
        donor_stimulus_id = stimulus_ids[group[assignment]], stratum = stratum,
        replicate = replicate, seed = seed, requested_replicates = replicates,
        eligible_donors = lengths(candidates),
        status = if (complete) "ok" else "no_valid_permutation", stringsAsFactors = FALSE)
    }
  }
  do.call(rbind, rows)
}

.saliency_transforms <- function(q) {
  out <- list(flip_axis1 = q[nrow(q):1, , drop = FALSE],
              flip_axis2 = q[, ncol(q):1, drop = FALSE],
              rotate180 = q[nrow(q):1, ncol(q):1, drop = FALSE])
  if (nrow(q) == ncol(q)) {
    out$rotate90 <- t(q[nrow(q):1, , drop = FALSE])
    out$rotate270 <- t(q[, ncol(q):1, drop = FALSE])
  }
  out
}

.saliency_controls <- function(q, status, centre_sigma, transformations,
                                donor_maps, assignments, image_id, representation,
                                signed_policy, original_shape, centre_match = FALSE,
                                donor_aspect_tolerance = .01) {
  shape <- dim(q)
  same_map <- function(a, b) identical(dim(a), dim(b)) && all(a == b)
  entry <- function(q, status, family, donor_id = NA_character_, replicate = NA_integer_,
                    metadata = list()) list(q = q, status = status, family = family,
      donor_id = donor_id, replicate = replicate, metadata = metadata)
  out <- list(observed = entry(q, status, "observed"),
              unweighted = entry(matrix(1 / length(q), shape[1], shape[2]), "ok", "unweighted"))
  if (!is.numeric(centre_sigma) || !length(centre_sigma) %in% c(1L, 2L) ||
      any(!is.finite(centre_sigma) | centre_sigma <= 0)) stop("centre_sigma must be one or two positive numbers")
  sigma <- rep(centre_sigma, length.out = 2L)
  x <- ((seq_len(shape[1]) - .5) / shape[1] - .5) / sigma[1]
  y <- ((seq_len(shape[2]) - .5) / shape[2] - .5) / sigma[2]
  log_centre <- -outer(x^2, y^2, "+") / 2
  centre <- .saliency_map(log_centre, "log_density")
  out$centre_bias <- entry(centre$q, centre$status, "centre_bias", metadata = list(
    source = "image-independent Gaussian", sigma_image_fraction = sigma))
  if (centre_match) {
    matched <- .saliency_match_centre(q, status)
    out$centre_bias_matched <- entry(matched$q, matched$status, "centre_bias_matched",
                                    metadata = matched$metadata)
  }
  if (transformations) {
    transformed <- .saliency_transforms(q)
    for (id in names(transformed)) out[[id]] <- entry(transformed[[id]], status, "transformed")
  }
  if (is.null(assignments)) {
    if (length(donor_maps)) stop("donor_maps require donor_assignments")
    out$donor_unavailable <- entry(matrix(0, shape[1], shape[2]), "no_donor_pool", "donor")
  } else {
    required <- c("target_id", "donor_id", "target_stimulus_id", "donor_stimulus_id", "replicate", "stratum", "status")
    if (!is.data.frame(assignments) || !all(required %in% names(assignments)))
      stop("donor_assignments must come from saliency_donor_assignments()")
    chosen <- assignments[assignments$target_id == image_id, , drop = FALSE]
    if (!nrow(chosen)) stop("No donor assignments for image_id")
    if (!is.list(donor_maps) || (length(donor_maps) &&
        (is.null(names(donor_maps)) || anyDuplicated(names(donor_maps)))))
      stop("donor_maps must be a uniquely named list")
    for (i in seq_len(nrow(chosen))) {
      row <- chosen[i, , drop = FALSE]
      id <- paste0("donor_", i)
      metadata <- as.list(row)
      if (row$status != "ok") {
        out[[id]] <- entry(matrix(0, shape[1], shape[2]), row$status, "donor",
                            row$donor_id, row$replicate, metadata)
        next
      }
      if (is.na(row$donor_id) || row$donor_id == image_id ||
          row$target_stimulus_id == row$donor_stimulus_id) stop("Self/duplicate-stimulus donors are not allowed")
      if (!row$donor_id %in% names(donor_maps)) stop("Missing donor map: ", row$donor_id)
      raw <- donor_maps[[row$donor_id]]
      donor <- .saliency_map(raw, representation, signed_policy)
      ratio_error <- abs((nrow(raw) / ncol(raw)) / (original_shape[1] / original_shape[2]) - 1)
      if (ratio_error > donor_aspect_tolerance)
        stop("Donor map aspect ratio must match the target")
      metadata$aspect_ratio_relative_error <- ratio_error
      metadata$aspect_ratio_tolerance <- donor_aspect_tolerance
      metadata$source_dimensions <- dim(raw)
      metadata$analysis_dimensions <- shape
      metadata$location_before_resize <- .saliency_location(donor$q, donor$status)
      metadata$resized <- !identical(as.integer(dim(raw)), as.integer(shape))
      metadata$map_interpolation <- .saliency_resize_method(dim(raw), shape)
      donor$q <- .saliency_resize(donor$q, shape)
      metadata$location_after_resize <- .saliency_location(donor$q, donor$status)
      out[[id]] <- entry(donor$q, donor$status, "donor", row$donor_id, row$replicate, metadata)
    }
  }
  for (i in seq_along(out)) {
    out[[i]]$identity_equivalent <- out[[i]]$status == "ok" && status == "ok" && same_map(out[[i]]$q, q)
    duplicates <- if (i > 1) which(vapply(out[seq_len(i - 1L)], function(x)
      x$status == "ok" && out[[i]]$status == "ok" && same_map(x$q, out[[i]]$q), logical(1))) else integer()
    out[[i]]$duplicate_of <- if (length(duplicates)) names(out)[duplicates[1]] else NA_character_
  }
  out
}

# Fit just the Shannon effective fraction, not the map's shape or edge support.
# A pixel-centred Gaussian on an even grid has multiple equal closest pixels:
# some target fractions are therefore unattainable and must remain unavailable.
.saliency_match_centre <- function(q, status) {
  shape <- dim(q)
  target <- .saliency_location(q, status)$effective_fraction
  radius <- outer(((seq_len(shape[1]) - (shape[1] + 1)/2) / shape[1])^2,
                  ((seq_len(shape[2]) - (shape[2] + 1)/2) / shape[2])^2, "+")
  radius <- radius - min(radius)
  gaussian <- function(log_sigma) .saliency_map(-radius / (2 * exp(2 * log_sigma)), "log_density")$q
  lower <- gaussian(log(1e-6))
  minimum <- .saliency_location(lower)$effective_fraction
  fit_status <- status
  sigma <- NA_real_
  fitted <- q * 0
  if (status == "ok") {
    if (target < minimum * (1 - 1e-8)) {
      fit_status <- "unattainable_effective_fraction"
      fitted <- lower
    } else if (target >= 1 - 1e-12) {
      fitted[] <- 1 / length(q)
      sigma <- Inf # Uniform limiting member of the Gaussian family.
    } else {
      objective <- function(s) .saliency_location(gaussian(s))$effective_fraction - target
      root <- if (target <= minimum) log(1e-6) else
        stats::uniroot(objective, c(log(1e-6), log(1e6)), tol = 1e-10)$root
      sigma <- exp(root)
      fitted <- gaussian(root)
    }
  }
  achieved <- if (any(fitted > 0)) .saliency_location(fitted)$effective_fraction else NA_real_
  list(q = fitted, status = fit_status, metadata = list(
    source = "effective-fraction-matched centred Gaussian", sigma_image_fraction = rep(sigma, 2),
    target_effective_fraction = target, achieved_effective_fraction = achieved,
    minimum_effective_fraction = minimum, fit_status = fit_status))
}
