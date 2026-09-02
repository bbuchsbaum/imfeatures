#' Linear-kernel CKA between two observation-by-feature matrices
#'
#' @param X,Y Numeric matrices with the same number of rows.
#' @return A scalar in \eqn{[0, 1]}, or \code{NA} if either matrix has no
#'   columns or a kernel is degenerate.
#' @export
linear_cka <- function(X, Y) {
  X <- as.matrix(X)
  Y <- as.matrix(Y)
  storage.mode(X) <- "double"
  storage.mode(Y) <- "double"
  if (nrow(X) != nrow(Y)) {
    stop("X and Y must have the same number of rows.")
  }
  if (ncol(X) == 0L || ncol(Y) == 0L) {
    return(NA_real_)
  }
  Xc <- scale(X, center = TRUE, scale = FALSE)
  Yc <- scale(Y, center = TRUE, scale = FALSE)
  xty <- crossprod(Xc, Yc)
  xtx <- crossprod(Xc)
  yty <- crossprod(Yc)
  num <- sum(xty * xty)
  den <- sqrt(sum(xtx * xtx) * sum(yty * yty))
  if (!is.finite(den) || den < .Machine$double.eps) {
    return(NA_real_)
  }
  as.numeric(num / den)
}

#' Spectral participation-ratio effective rank
#'
#' @param X Numeric matrix.
#' @param center Logical; centre columns before the SVD.
#' @return A non-negative scalar.
#' @export
effective_rank <- function(X, center = TRUE) {
  X <- as.matrix(X)
  storage.mode(X) <- "double"
  if (isTRUE(center)) {
    X <- scale(X, center = TRUE, scale = FALSE)
  }
  if (ncol(X) == 0L || nrow(X) < 2L) {
    return(0)
  }
  d <- svd(X, nu = 0, nv = 0)$d
  d2 <- d^2
  s <- sum(d2)
  if (!is.finite(s) || s < .Machine$double.eps) {
    return(0)
  }
  as.numeric((s^2) / sum(d2^2))
}

#' Stimulus-only diagnostics for a fitted hierarchy transform
#'
#' @param fit A \code{hierarchy_transform}.
#' @param blocks Blocks used to compute in-sample or supplied-sample
#'   diagnostics. Typically the training blocks.
#' @param heldout Optional held-out blocks for residual CKA and residual
#'   energy.
#'
#' @return An object of class \code{hierarchy_diagnostics}.
#' @export
hierarchy_diagnostics <- function(fit, blocks, heldout = NULL) {
  if (!inherits(fit, "hierarchy_transform")) {
    stop("'fit' must be a hierarchy_transform.")
  }
  reps <- .hierarchy_representations(fit, blocks)
  compression <- .compression_table(fit, reps)
  innovation <- .innovation_table(reps)
  pairwise <- .pairwise_cka(reps$E)
  held <- NULL
  if (!is.null(heldout)) {
    held_reps <- .hierarchy_representations(fit, heldout)
    held <- list(
      residual_energy = mapply(
        function(E, Z) .frobenius_ratio(E, Z),
        held_reps$E, held_reps$Z
      ),
      pairwise_cka = .pairwise_cka(held_reps$E)
    )
  }
  structure(
    list(
      compression = compression,
      innovation = innovation,
      pairwise_cka = pairwise,
      heldout = held
    ),
    class = "hierarchy_diagnostics"
  )
}

#' Check batch invariance and row-permutation equivariance
#'
#' @param fit A \code{hierarchy_transform}.
#' @param blocks Blocks to transform.
#' @param idx Optional row index used for the batch-invariance check.
#'   Defaults to a deterministic subset.
#' @param tolerance Absolute tolerance for \code{all.equal}.
#' @return A list with logical flags and maximum absolute differences.
#' @export
check_transform_invariance <- function(fit, blocks, idx = NULL, tolerance = 1e-8) {
  pred_all <- predict(fit, blocks)
  n <- nrow(pred_all[[1]])
  if (is.null(idx)) {
    idx <- unique(as.integer(seq(1L, n, length.out = min(n, max(2L, floor(n / 3))))))
  }
  pred_sub <- predict(fit, subset_feature_blocks(blocks, idx))
  batch_diffs <- vapply(names(pred_all), function(nm) {
    max(abs(pred_all[[nm]][idx, , drop = FALSE] - pred_sub[[nm]]))
  }, numeric(1))
  perm <- if (n >= 2L) c(2L, 1L, if (n > 2L) seq.int(3L, n)) else 1L
  pred_perm <- predict(fit, subset_feature_blocks(blocks, perm))
  perm_diffs <- vapply(names(pred_all), function(nm) {
    max(abs(pred_perm[[nm]] - pred_all[[nm]][perm, , drop = FALSE]))
  }, numeric(1))
  list(
    batch_invariant = all(batch_diffs <= tolerance | !is.finite(batch_diffs)),
    permutation_equivariant = all(perm_diffs <= tolerance | !is.finite(perm_diffs)),
    max_batch_diff = max(batch_diffs, na.rm = TRUE),
    max_perm_diff = max(perm_diffs, na.rm = TRUE)
  )
}

#' @export
print.hierarchy_diagnostics <- function(x, ...) {
  cat("Hierarchy diagnostics\n")
  if (!is.null(x$compression)) {
    cat("Compression:\n")
    print(x$compression, row.names = FALSE)
  }
  if (!is.null(x$innovation)) {
    cat("Innovation:\n")
    print(x$innovation, row.names = FALSE)
  }
  invisible(x)
}

#' @keywords internal
.frobenius_ratio <- function(num, den) {
  num <- as.matrix(num)
  den <- as.matrix(den)
  dn <- sum(den * den)
  if (!is.finite(dn) || dn < .Machine$double.eps) {
    return(NA_real_)
  }
  as.numeric(sum(num * num) / dn)
}

#' @keywords internal
.hierarchy_representations <- function(fit, blocks) {
  block_list <- .as_block_list(blocks, arg = "blocks")
  stages <- fit$stage_order
  n <- nrow(block_list[[stages[[1]]]])
  X <- block_list[stages]
  Z <- vector("list", length(stages))
  names(Z) <- stages
  for (st in stages) {
    Z[[st]] <- .apply_block_pca(fit$compress[[st]], X[[st]], st)
  }
  E <- Z
  if (!is.null(fit$recipe$decomposition)) {
    prev <- list()
    for (st in stages) {
      H <- .bind_score_blocks(prev, n)
      E[[st]] <- .apply_stage_mapping(H, Z[[st]], fit$mapping[[st]]$B_hat)
      prev[[st]] <- Z[[st]]
    }
  }
  Xhat <- vector("list", length(stages))
  names(Xhat) <- stages
  for (st in stages) {
    Xhat[[st]] <- .reconstruct_from_pca(fit$compress[[st]], Z[[st]])
  }
  list(X = X, Z = Z, E = E, Xhat = Xhat)
}

#' @keywords internal
.compression_table <- function(fit, reps) {
  stages <- fit$stage_order
  do.call(rbind, lapply(stages, function(st) {
    X <- reps$X[[st]]
    Z <- reps$Z[[st]]
    Xhat <- reps$Xhat[[st]]
    data.frame(
      stage = st,
      dim_in = ncol(X),
      dim_out = ncol(Z),
      variance_retained = 1 - {
        xc <- scale(X, center = TRUE, scale = FALSE)
        rc <- X - Xhat
        rc <- scale(rc, center = TRUE, scale = FALSE)
        .frobenius_ratio(rc, xc)
      },
      cka_retained = linear_cka(X, Xhat),
      effective_rank = effective_rank(Z),
      recon_error = {
        xc <- scale(X, center = TRUE, scale = FALSE)
        rc <- scale(X - Xhat, center = TRUE, scale = FALSE)
        .frobenius_ratio(rc, xc)
      },
      stringsAsFactors = FALSE
    )
  }))
}

#' @keywords internal
.innovation_table <- function(reps) {
  stages <- names(reps$Z)
  do.call(rbind, lapply(seq_along(stages), function(i) {
    st <- stages[[i]]
    energy <- .frobenius_ratio(reps$E[[st]], reps$Z[[st]])
    data.frame(
      stage = st,
      residual_energy = energy,
      fraction_predictable = if (is.na(energy)) NA_real_ else 1 - energy,
      n_components = ncol(reps$E[[st]]),
      stringsAsFactors = FALSE
    )
  }))
}

#' @keywords internal
.pairwise_cka <- function(mats) {
  nms <- names(mats)
  out <- matrix(NA_real_, length(nms), length(nms), dimnames = list(nms, nms))
  for (i in seq_along(nms)) {
    for (j in seq_along(nms)) {
      out[i, j] <- linear_cka(mats[[i]], mats[[j]])
    }
  }
  out
}
