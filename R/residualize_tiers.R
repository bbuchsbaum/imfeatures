#' Hierarchical residualization of tiered feature matrices with PCA
#'
#' Given a named list of feature matrices (e.g., low-, mid-, high-tier features),
#' perform PCA within each tier to reduce to a specified number of components,
#' then sequentially residualize each tier's PCs against all previous tiers using
#' an SVD-based rank-aware approach. Optionally, PC scores can be z-scored.
#'
#' @param feature_list Named list of numeric matrices, each of dimension N_samples × P_i.
#' @param numpcs Integer scalar or numeric vector, or NULL. If NULL (default), uses up to 50 PCs per tier (or fewer, if a tier has <50 features).
#' @param pca_method Character string, one of "stats" (default, uses \code{stats::prcomp}) or "irlba" (uses \code{irlba::prcomp_irlba}).
#' @param svd_tol Numeric tolerance factor used in determining effective rank via SVD for sequential residualization. 
#'        Singular values \code{s_i} are considered non-zero if \code{s_i > svd_tol * max(dim(H)) * s[1]},
#'        where \code{H} is the matrix being decomposed and \code{s[1]} is its largest singular value.
#'        Default is 1e-7.
#' @param scale_scores Logical. If TRUE (default), PC scores for each tier are z-scored (scaled to unit standard deviation, no centering as PCA already centers) 
#'        before residualization. SDs for scaling in \code{predict} are taken from the training data.
#' @return An object of class \code{residualized_tiers}, a list with components:
#' \describe{
#'   \item{pca}{Named list of PCA objects. If \code{scale_scores = TRUE}, this will also contain \code{sds_for_scaling} for each tier.}
#'   \item{pc_scores_raw}{Named list of raw PC score matrices (N_samples × numpcs[i]) before residualization (but after optional scaling if \code{scale_scores=TRUE}).}
#'   \item{residuals}{Named list of final residualized PC matrices.}
#'   \item{projection_bases}{Named list. For each tier (except the first), stores the orthonormal basis matrix Q (from SVD of preceding tiers' cumulative non-residualized PC scores, truncated by rank), or NULL.}
#'   \item{numpcs}{Integer vector of number of PCs per tier.}
#'   \item{tiers}{Character vector of tier names.}
#'   \item{svd_tol_info}{List containing the \code{svd_tol} value used and a description of the tolerance formula.}
#'   \item{scale_scores}{Logical value of the \code{scale_scores} argument used.}
#' }
#' An attribute \code{total_rank_kept} (sum of \code{numpcs}) is also attached.
#' @details
#' Initial PCA is performed on each tier. If \code{scale_scores = TRUE}, the resulting PC scores are then z-scored (column-wise, per tier).
#' These (optionally scaled) PC scores are then sequentially residualized.
#' The orthogonalization uses SVD to form a rank-aware basis of the preceding tiers' cumulative scores 
#' to ensure numerical stability, using the \code{svd_tol} parameter in a LAPACK-style manner.
#' Note: For applications like fMRI analysis where data might be processed in folds (e.g., for cross-validation),
#' z-scoring should ideally be performed based on training fold statistics and then applied to test folds.
#' This function, when applied to a whole dataset, uses statistics from the entire input for z-scoring.
#'
#' @export
residualize_tiers <- function(feature_list, numpcs = NULL, 
                              pca_method = c("stats", "irlba"), 
                              svd_tol = 1e-7,
                              scale_scores = TRUE) {
  if (!is.list(feature_list) || is.null(names(feature_list))) {
    stop("feature_list must be a named list of matrices.")
  }
  # Ensure all matrices have the same number of rows (samples)
  row_counts <- vapply(feature_list, function(x) nrow(as.matrix(x)), integer(1))
  if (length(unique(row_counts)) != 1) {
    counts_str <- paste(sprintf("%s=%d", names(row_counts), row_counts), collapse = ", ")
    stop(sprintf("All matrices in 'feature_list' must have the same number of rows (samples); got: %s", counts_str))
  }
  pca_method <- match.arg(pca_method)

  if (!is.numeric(svd_tol) || svd_tol <= 0) {
    stop("svd_tol must be a positive numeric value.")
  }
  if (!is.logical(scale_scores) || length(scale_scores) != 1) {
    stop("scale_scores must be a single logical value (TRUE or FALSE).")
  }

  ntiers <- length(feature_list)
  tier_names <- names(feature_list)

  if (is.null(numpcs)) {
    Pvec <- vapply(feature_list, function(X) ncol(as.matrix(X)), integer(1))
    numpcs <- pmin(50L, Pvec)
  }
  numpcs <- as.integer(numpcs)
  if (length(numpcs) == 1) {
    numpcs <- rep(numpcs, ntiers)
  } else if (length(numpcs) != ntiers) {
    stop("numpcs must be NULL, length 1, or length equal to number of tiers (", ntiers, ").")
  }

  pca_list <- vector("list", ntiers)
  pc_scores_raw <- vector("list", ntiers) 
  names(pca_list) <- tier_names
  names(pc_scores_raw) <- tier_names
  numpcs_computed <- numpcs 

  for (i in seq_along(feature_list)) {
    tier_name <- tier_names[i]
    X <- feature_list[[i]]
    if (!is.matrix(X)) {
      X <- as.matrix(X)
    }
    k_requested <- numpcs[i]
    P <- ncol(X)
    N_samples <- nrow(X)

    if (N_samples < 2 && k_requested > 0) {
        stop(sprintf("Tier '%s': Need >=2 samples for PCA to compute positive PCs; got N_samples=%d for k_requested=%d", tier_name, N_samples, k_requested))
    }

    k <- min(k_requested, P, if(N_samples > 0) N_samples - 1 else 0, na.rm = TRUE)
    
    current_pca_method <- pca_method

    if (k <= 0 && k_requested > 0) { 
      stop(sprintf("Number of PCs to keep (k_computed=%d, from k_requested=%d) must be positive for tier '%s'. Check numpcs or data dimensions (N=%d, P=%d).", k, k_requested, tier_name, N_samples, P))
    } else if (k <= 0 && k_requested <= 0) { 
       if (k_requested < 0) warning(sprintf("Tier '%s': Requested k=%d PCs, will use 0 PCs.", tier_name, k_requested))
       warning(sprintf("Tier '%s': Effective k=0 PCs. No PCA will be performed. Scores will be a 0-column matrix.", tier_name))
       pca_obj <- list(center = if(P>0) rep(0, P) else numeric(0), 
                       scale = if(P>0) rep(1, P) else numeric(0), 
                       rotation = matrix(0, P, 0), 
                       x = matrix(0, N_samples, 0))
       if(P > 0 && !is.null(colnames(X))) { names(pca_obj$center) <- names(pca_obj$scale) <- colnames(X) }
       k_computed_pca <- 0
       if(scale_scores) pca_obj$sds_for_scaling <- numeric(0)
    } else { 
        if (current_pca_method == "irlba") {
          if (requireNamespace("irlba", quietly = TRUE)) {
            if (k >= min(N_samples, P) && min(N_samples,P) > 1) {
                 warning(sprintf("Tier '%s': k=%d is close to full rank (min(N,P)=%d). Using stats::prcomp for stability/completeness instead of irlba.", 
                                tier_name, k, min(N_samples,P)))
                 current_pca_method <- "stats"
            } 
          } else {
            warning("pca_method = 'irlba' was chosen, but 'irlba' package is not installed. Falling back to pca_method = 'stats'.")
            current_pca_method <- "stats"
          }
        }

        if (current_pca_method == "stats"){
             pca_obj <- stats::prcomp(X, center = TRUE, scale. = TRUE, retx = TRUE, rank. = k) 
        } else { 
             pca_obj <- irlba::prcomp_irlba(X, n = k, center = TRUE, scale. = TRUE)
        }
        
        if (ncol(pca_obj$x) < k) {
            warning(sprintf("Tier '%s': Requested k=%d PCs, but PCA computed %d PCs. Using %d PCs.", 
                            tier_name, k, ncol(pca_obj$x), ncol(pca_obj$x)))
            k_computed_pca <- ncol(pca_obj$x)
        } else {
            k_computed_pca <- k
        }
        
        if (scale_scores && k_computed_pca > 0) {
            sds_train <- apply(pca_obj$x[, seq_len(k_computed_pca), drop = FALSE], 2, stats::sd)
            pca_obj$sds_for_scaling <- sds_train
        }
    }

    pca_list[[tier_name]] <- pca_obj
    numpcs_computed[i] <- k_computed_pca
    
    current_tier_scores <- if (k_computed_pca > 0) {
        pca_obj$x[, seq_len(k_computed_pca), drop = FALSE]
    } else {
        matrix(0.0, nrow = N_samples, ncol = 0)
    }

    if (scale_scores && k_computed_pca > 0) {
        sds_to_use <- pca_obj$sds_for_scaling
        pc_scores_raw[[tier_name]] <- scale(current_tier_scores, center = FALSE, scale = sds_to_use + 1e-8)
    } else {
        pc_scores_raw[[tier_name]] <- current_tier_scores
    }
  }
  numpcs <- numpcs_computed

  residuals_final <- vector("list", ntiers)
  projection_bases <- vector("list", ntiers)
  names(residuals_final) <- tier_names
  names(projection_bases) <- tier_names
  
  H_cumulative <- NULL
  for (i in seq_along(tier_names)) {
    tier_name <- tier_names[i]
    current_scores <- pc_scores_raw[[tier_name]]

    if (is.null(H_cumulative) || ncol(H_cumulative) == 0) {
      residuals_final[[tier_name]] <- current_scores
      projection_bases[[tier_name]] <- NULL 
    } else {
      num_left_vectors <- min(nrow(H_cumulative), ncol(H_cumulative))
      svd_H <- svd(H_cumulative, nu = num_left_vectors, nv = 0)
      
      abs_rank_tol <- svd_tol * max(dim(H_cumulative)) * svd_H$d[1]
      if (svd_H$d[1] < .Machine$double.eps) { 
          effective_rank <- 0
      } else {
          effective_rank <- sum(svd_H$d > abs_rank_tol & svd_H$d > .Machine$double.eps)
      }
              
      if (effective_rank > 0) {
        Q <- svd_H$u[, seq_len(effective_rank), drop = FALSE]
        projection_bases[[tier_name]] <- Q
        proj_coeffs <- crossprod(Q, current_scores)
        residuals_final[[tier_name]] <- current_scores - (Q %*% proj_coeffs)
      } else { 
        residuals_final[[tier_name]] <- current_scores
        projection_bases[[tier_name]] <- NULL
      }
    }
    if (ncol(current_scores) > 0) {
        if (is.null(H_cumulative) || ncol(H_cumulative) == 0) { 
            H_cumulative <- current_scores
        } else {
            H_cumulative <- cbind(H_cumulative, current_scores)
        }
    } 
  }

  result <- list(
    pca = pca_list,
    pc_scores_raw = pc_scores_raw, 
    residuals = residuals_final,
    projection_bases = projection_bases,
    numpcs = numpcs,
    tiers = tier_names,
    svd_tol_info = list(value = svd_tol, formula = "s_i > svd_tol * max(dim(H)) * s[1]"),
    scale_scores = scale_scores
  )
  class(result) <- "residualized_tiers"
  attr(result, "total_rank_kept") <- sum(result$numpcs)
  result
}

#' @export
print.residualized_tiers <- function(x, ...) {
  cat("Residualized tiered features (Sequential SVD-QR method)\n")
  cat("  Tiers: ", paste(x$tiers, collapse = ", "), "\n")
  cat("  Num PCs per tier (computed): ", paste(x$numpcs, collapse = ", "), "\n")
  cat("  Total rank kept: ", sum(x$numpcs), "\n")
  cat("  Scores scaled (z-scored by tier): ", x$scale_scores, "\n")
  if (!is.null(x$svd_tol_info)) {
    cat("  SVD tolerance factor: ", x$svd_tol_info$value, 
        " (applied as ", x$svd_tol_info$formula, ")\n")
  }
  invisible(x)
}

#' Predict method for residualized_tiers
#'
#' Applies a trained residualized_tiers transformation (Sequential SVD-QR method) to new data.
#' @method predict residualized_tiers
#' @export
predict.residualized_tiers <- function(object, newdata, ...) {
  if (!inherits(object, "residualized_tiers")) {
    stop("Input 'object' must be of class 'residualized_tiers'.")
  }
  if (!is.list(newdata) || is.null(names(newdata))) {
    stop("'newdata' must be a named list of matrices.")
  }
  if (!identical(sort(names(newdata)), sort(object$tiers))) {
    stop("'newdata' must contain matrices for all tiers present in the training object: ",
         paste(object$tiers, collapse = ", "))
  }
  num_samples_new <- nrow(as.matrix(newdata[[object$tiers[1]]]))
  if (num_samples_new == 0) stop("'newdata' matrices cannot have zero rows.")
  if (length(object$tiers) > 1) {
    for (tier_idx in 2:length(object$tiers)) {
      if (nrow(as.matrix(newdata[[object$tiers[tier_idx]]])) != num_samples_new) {
        stop("All matrices in 'newdata' must have the same number of rows (samples).")
      }
    }
  }

  ntiers <- length(object$tiers)
  new_pc_scores_raw_list <- vector("list", ntiers)
  names(new_pc_scores_raw_list) <- object$tiers
  scale_scores_train <- object$scale_scores

  for (i in seq_along(object$tiers)) {
    tier_name <- object$tiers[i]
    current_X_new <- as.matrix(newdata[[tier_name]])
    pca_obj_train <- object$pca[[tier_name]]
    k_train <- object$numpcs[i] 

    if (is.null(pca_obj_train$center)) {
        if (k_train != 0) stop(sprintf("Mismatch: k_train is %d for tier '%s' but PCA object seems empty.", k_train, tier_name))
    } else if (ncol(current_X_new) != length(pca_obj_train$center)) {
      stop(sprintf("Column count mismatch for tier '%s' in 'newdata'. Expected %d, got %d.",
                   tier_name, length(pca_obj_train$center), ncol(current_X_new)))
    }
    training_colnames <- names(pca_obj_train$center)
    new_data_colnames <- colnames(current_X_new)
    if (!is.null(training_colnames) && !is.null(new_data_colnames)) {
      if (!identical(new_data_colnames, training_colnames)) {
        if (all(training_colnames %in% new_data_colnames) && all(new_data_colnames %in% training_colnames)) {
          current_X_new <- current_X_new[, training_colnames, drop = FALSE]
        } else {
          stop(sprintf("Column names for tier '%s' in 'newdata' do not match or are not a permutation of training data column names.", tier_name))
        }
      }
    } else if (is.null(training_colnames) && !is.null(new_data_colnames)) {
      warning(sprintf("Tier '%s' training data had no column names, but newdata does. Proceeding by column index.", tier_name))
    } else if (!is.null(training_colnames) && is.null(new_data_colnames)) {
      stop(sprintf("Tier '%s' training data had column names, but newdata does not. Cannot ensure correct column order.", tier_name))
    }
    
    current_tier_new_scores <- if (k_train == 0) { 
        matrix(0.0, nrow = num_samples_new, ncol = 0)
    } else {
        pred_pcs <- predict(pca_obj_train, newdata = current_X_new)
        pred_pcs[, seq_len(min(k_train, ncol(pred_pcs))), drop = FALSE]
    }

    if (scale_scores_train && k_train > 0 && !is.null(pca_obj_train$sds_for_scaling)) {
        new_pc_scores_raw_list[[tier_name]] <- scale(current_tier_new_scores, center = FALSE, scale = pca_obj_train$sds_for_scaling + 1e-8)
    } else {
        new_pc_scores_raw_list[[tier_name]] <- current_tier_new_scores
    }
  }

  residuals_new_final <- vector("list", ntiers)
  names(residuals_new_final) <- object$tiers
  
  current_svd_tol <- if (!is.null(object$svd_tol_info) && !is.null(object$svd_tol_info$value)) {
    object$svd_tol_info$value
  } else { 
    1e-7 
  }

  H_new_cumulative <- NULL
  for (i in seq_along(object$tiers)) {
    tier_name <- object$tiers[i]
    current_new_scores <- new_pc_scores_raw_list[[tier_name]]

    if (is.null(H_new_cumulative) || ncol(H_new_cumulative) == 0) {
      residuals_new_final[[tier_name]] <- current_new_scores
    } else {
      num_lv_new <- min(nrow(H_new_cumulative), ncol(H_new_cumulative))
      svd_H_new  <- svd(H_new_cumulative, nu = num_lv_new, nv = 0)
      
      abs_rank_tol_new <- current_svd_tol * max(dim(H_new_cumulative)) * svd_H_new$d[1]
      if (svd_H_new$d[1] < .Machine$double.eps) {
          effective_rank_new <- 0
      } else {
          effective_rank_new <- sum(svd_H_new$d > abs_rank_tol_new & svd_H_new$d > .Machine$double.eps)
      }
            
      if (effective_rank_new > 0) {
        Q_basis_new_data <- svd_H_new$u[, seq_len(effective_rank_new), drop = FALSE]
        proj_coeffs_new <- crossprod(Q_basis_new_data, current_new_scores)
        residuals_new_final[[tier_name]] <- current_new_scores - (Q_basis_new_data %*% proj_coeffs_new)
      } else { 
        residuals_new_final[[tier_name]] <- current_new_scores
      }
    }
    if (ncol(current_new_scores) > 0) {
        if (is.null(H_new_cumulative) || ncol(H_new_cumulative) == 0) {
            H_new_cumulative <- current_new_scores
        } else {
            H_new_cumulative <- cbind(H_new_cumulative, current_new_scores)
        }
    }
  }
  
  return(residuals_new_final)
} 