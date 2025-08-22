#' Sequential orthogonal residualization of score matrices
#'
#' This helper takes a list of score matrices (each with the same number of rows)
#' and sequentially residualizes each matrix against the cumulative scores of all
#' previous matrices using an SVD-based rank-aware approach.
#'
#' @param scores_list Named list of numeric matrices.
#' @param svd_tol Numeric tolerance factor for determining effective rank in the
#'   SVD step.
#' @param return_projection_bases Logical; return the orthonormal projection bases
#'   used at each step?
#' @return If \code{return_projection_bases = TRUE}, a list with elements
#'   \code{residuals} and \code{projection_bases}. Otherwise, just the list of
#'   residual matrices.
#' @keywords internal
.orthogonal_residuals <- function(scores_list, svd_tol,
                                  return_projection_bases = TRUE) {
  n <- length(scores_list)
  residuals_list <- vector("list", n)
  if (return_projection_bases) {
    proj_bases <- vector("list", n)
  } else {
    proj_bases <- NULL
  }
  names(residuals_list) <- names(scores_list)
  if (!is.null(proj_bases)) names(proj_bases) <- names(scores_list)

  H_cumulative <- NULL
  for (i in seq_along(scores_list)) {
    current_scores <- scores_list[[i]]

    if (is.null(H_cumulative) || ncol(H_cumulative) == 0) {
      residuals_list[[i]] <- current_scores
      if (!is.null(proj_bases)) proj_bases[[i]] <- NULL
    } else {
      num_lv <- min(nrow(H_cumulative), ncol(H_cumulative))
      svd_H  <- svd(H_cumulative, nu = num_lv, nv = 0)

      abs_rank_tol <- svd_tol * max(dim(H_cumulative)) * svd_H$d[1]
      if (svd_H$d[1] < .Machine$double.eps) {
        effective_rank <- 0
      } else {
        effective_rank <- sum(svd_H$d > abs_rank_tol &
                              svd_H$d > .Machine$double.eps)
      }

      if (effective_rank > 0) {
        Q <- svd_H$u[, seq_len(effective_rank), drop = FALSE]
        if (!is.null(proj_bases)) proj_bases[[i]] <- Q
        proj_coeffs <- crossprod(Q, current_scores)
        residuals_list[[i]] <- current_scores - (Q %*% proj_coeffs)
      } else {
        residuals_list[[i]] <- current_scores
        if (!is.null(proj_bases)) proj_bases[[i]] <- NULL
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

  if (return_projection_bases) {
    list(residuals = residuals_list, projection_bases = proj_bases)
  } else {
    residuals_list
  }
}
