#' Hierarchical residualization of tiered feature matrices with PCA
#'
#' Compatibility wrapper around \code{\link{fit_hierarchy_transform}}. Given a
#' named list of feature matrices, it compresses each tier with PCA and
#' residualizes later tiers against earlier tiers using a stored cross-tier
#' mapping. New observations are transformed inductively: each row depends
#' only on that row's own tier values and the training-fitted maps.
#'
#' @param feature_list Named list of numeric matrices, each of dimension N_samples × P_i.
#' @param numpcs Integer scalar or numeric vector, or NULL. If NULL (default), uses up to 50 PCs per tier (or fewer, if a tier has <50 features).
#' @param pca_method Character string, one of "stats" (default, uses \code{stats::prcomp}) or "eigencore" (uses \code{eigencore::svd_partial}).
#' @param svd_tol Numeric tolerance factor used in determining effective rank via SVD for sequential residualization.
#'        Singular values \code{s_i} are considered non-zero if \code{s_i > svd_tol * max(dim(H)) * s[1]},
#'        where \code{H} is the matrix being decomposed and \code{s[1]} is its largest singular value.
#'        Default is 1e-7.
#' @param scale_scores Logical. If TRUE (default), PC scores for each tier are z-scored (scaled to unit standard deviation, no centering as PCA already centers)
#'        before residualization. SDs for scaling in \code{predict} are taken from the training data.
#' @return An object of class \code{c("residualized_tiers", "hierarchy_transform")}, a list with components:
#' \describe{
#'   \item{pca}{Named list of PCA objects. If \code{scale_scores = TRUE}, this will also contain \code{sds_for_scaling} for each tier.}
#'   \item{pc_scores_raw}{Named list of raw PC score matrices (N_samples × numpcs[i]) before residualization (but after optional scaling if \code{scale_scores=TRUE}).}
#'   \item{residuals}{Named list of final residualized PC matrices.}
#'   \item{projection_bases}{Named list. For each tier (except the first), the training-set orthonormal basis of preceding scores, stored as a diagnostic only. Prediction uses stored coefficient maps, not these bases.}
#'   \item{numpcs}{Integer vector of number of PCs per tier.}
#'   \item{tiers}{Character vector of tier names.}
#'   \item{svd_tol_info}{List containing the \code{svd_tol} value used and a description of the tolerance formula.}
#'   \item{scale_scores}{Logical value of the \code{scale_scores} argument used.}
#' }
#' An attribute \code{total_rank_kept} (sum of \code{numpcs}) is also attached.
#' @details
#' Initial PCA is performed on each tier. If \code{scale_scores = TRUE}, the resulting PC scores are then z-scored (column-wise, per tier).
#' These (optionally scaled) PC scores are then sequentially residualized.
#' Residualization stores the least-squares map from earlier-tier scores to the
#' current tier, so \code{predict()} does not re-estimate that map from the
#' new batch. Exact orthogonality is expected on the training observations for
#' unregularized residualization and is not expected on new data.
#' All matrices must contain only finite values; the function stops if any NA or Inf values are detected.
#'
#' @export
residualize_tiers <- function(feature_list, numpcs = NULL,
                              pca_method = c("stats", "eigencore"),
                              svd_tol = 1e-7,
                              scale_scores = TRUE) {
  feature_list <- .validate_block_list(feature_list, arg = "feature_list", label = "Tier")
  pca_method <- match.arg(pca_method)
  checkmate::assert_number(svd_tol, lower = 0)
  assert_scalar(scale_scores, "logical")

  if (!is.null(numpcs)) {
    checkmate::assert_numeric(numpcs, lower = 0, finite = TRUE, any.missing = FALSE)
    if (length(numpcs) != 1L && length(numpcs) != length(feature_list)) {
      stop(
        "numpcs must be NULL, length 1, or length equal to number of tiers (",
        length(feature_list), ")."
      )
    }
  }

  recipe <- hierarchy_recipe(
    order = names(feature_list),
    compress = block_pca(
      rank = numpcs,
      method = pca_method,
      feature_scale = "sd",
      score_weighting = if (scale_scores) "whitened" else "singular"
    ),
    decomposition = ordered_innovation(
      method = "svd",
      tolerance = svd_tol
    ),
    innovation_suffix = FALSE
  )
  fit <- fit_hierarchy_transform(recipe, feature_list)
  .as_residualized_tiers(fit, scale_scores = scale_scores, svd_tol = svd_tol)
}

#' @export
print.residualized_tiers <- function(x, ...) {
  cat("Residualized tiered features (inductive SVD mapping)\n")
  cat("  Tiers: ", paste(x$tiers, collapse = ", "), "\n")
  cat("  Num PCs per tier (computed): ", paste(x$numpcs, collapse = ", "), "\n")
  cat("  Total rank kept: ", sum(x$numpcs), "\n")
  cat("  Scores scaled (z-scored by tier): ", x$scale_scores, "\n")
  if (!is.null(x$svd_tol_info)) {
    cat(
      "  SVD tolerance factor: ", x$svd_tol_info$value,
      " (applied as ", x$svd_tol_info$formula, ")\n"
    )
  }
  invisible(x)
}

#' Predict method for residualized_tiers
#'
#' Applies a trained residualized_tiers transformation to new data using the
#' stored PCA maps and cross-tier coefficient matrices.
#'
#' @param object An object of class \code{residualized_tiers} produced by \code{residualize_tiers()}.
#' @param newdata Named list of matrices with the same tier names and feature columns as the training data.
#' @param ... Additional arguments (currently ignored).
#' @return Named list of residualized PC matrices for each tier in \code{newdata}.
#' @method predict residualized_tiers
#' @export
predict.residualized_tiers <- function(object, newdata, ...) {
  if (!inherits(object, "residualized_tiers")) {
    stop("Input 'object' must be of class 'residualized_tiers'.")
  }
  if (!is.list(newdata) || is.null(names(newdata))) {
    if (!inherits(newdata, "feature_hierarchy")) {
      stop("'newdata' must be a named list of matrices.")
    }
  }
  predict.hierarchy_transform(object, newdata, ...)
}

#' @keywords internal
.as_residualized_tiers <- function(fit, scale_scores, svd_tol) {
  stages <- fit$stage_order
  pca_list <- vector("list", length(stages))
  names(pca_list) <- stages
  for (st in stages) {
    cf <- fit$compress[[st]]
    pca <- cf$prcomp
    if (is.null(pca)) {
      pca <- list(
        center = cf$center,
        scale = cf$scale,
        rotation = cf$rotation,
        x = cf$scores_unweighted
      )
    }
    if (scale_scores) {
      pca$sds_for_scaling <- cf$score_sds
    }
    pca_list[[st]] <- pca
  }

  residuals <- fit$residuals
  names(residuals) <- stages
  for (st in stages) {
    colnames(residuals[[st]]) <- colnames(fit$residuals[[fit$output_names[[st]]]])
  }

  orth <- .orthogonal_residuals(
    fit$training_scores,
    svd_tol,
    return_projection_bases = TRUE
  )

  fit$pca <- pca_list
  fit$pc_scores_raw <- fit$training_scores
  fit$residuals <- residuals
  fit$projection_bases <- orth$projection_bases
  fit$numpcs <- vapply(fit$compress[stages], function(cf) as.integer(cf$rank), integer(1))
  names(fit$numpcs) <- NULL
  fit$tiers <- stages
  fit$svd_tol_info <- list(
    value = svd_tol,
    formula = "s_i > svd_tol * max(dim(H)) * s[1]"
  )
  fit$scale_scores <- scale_scores
  class(fit) <- c("residualized_tiers", "hierarchy_transform")
  attr(fit, "total_rank_kept") <- sum(fit$numpcs)
  fit
}
