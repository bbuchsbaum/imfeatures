HIERARCHY_TRANSFORM_VERSION <- "1.0.0"

#' Fit an inductive hierarchy transformation
#'
#' Learns per-stage compression, cross-stage residualization maps, and
#' optional post-residual compression from training blocks. The fitted object
#' stores coefficient matrices, not observation-space projection bases, so
#' new rows are transformed independently of batch membership.
#'
#' @param recipe A \code{\link{hierarchy_recipe}}.
#' @param blocks A named list of numeric matrices or a
#'   \code{\link{as_feature_hierarchy}} object. Names must include every
#'   stage in \code{recipe$order}.
#' @param block_id Optional blocking factor reserved for future
#'   \code{lambda = "blocked_cv"} support. Currently unused.
#'
#' @return An object of class \code{hierarchy_transform}.
#'
#' @examples
#' set.seed(1)
#' blocks <- list(
#'   b1 = matrix(rnorm(80), 40, 2, dimnames = list(NULL, c("a", "b"))),
#'   b2 = matrix(rnorm(120), 40, 3, dimnames = list(NULL, c("x", "y", "z")))
#' )
#' recipe <- hierarchy_recipe(
#'   order = c("b1", "b2"),
#'   compress = block_pca(rank = 2),
#'   decomposition = ordered_innovation(method = "svd")
#' )
#' fit <- fit_hierarchy_transform(recipe, blocks)
#' transformed <- predict(fit, blocks)
#' @export
fit_hierarchy_transform <- function(recipe, blocks, block_id = NULL) {
  if (!inherits(recipe, "hierarchy_recipe")) {
    stop("'recipe' must be created by hierarchy_recipe().")
  }
  if (!is.null(recipe$decomposition) &&
    identical(recipe$decomposition$lambda, "blocked_cv")) {
    stop(
      "ordered_innovation(lambda = \"blocked_cv\") is not implemented yet.",
      call. = FALSE
    )
  }
  if (!is.null(block_id)) {
    # Reserved for blocked_cv; accepted so callers can already pass scene IDs.
    invisible(block_id)
  }

  block_list <- .as_block_list(blocks, arg = "blocks")
  stages <- recipe$order
  missing <- setdiff(stages, names(block_list))
  extra <- setdiff(names(block_list), stages)
  if (length(missing)) {
    stop(
      "'blocks' is missing stages required by the recipe: ",
      paste(missing, collapse = ", "),
      "."
    )
  }
  if (length(extra)) {
    block_list <- block_list[stages]
  } else {
    block_list <- block_list[stages]
  }

  n <- nrow(block_list[[1]])
  n_features <- vapply(block_list, ncol, integer(1))
  compress_ranks <- .resolve_stage_ranks(
    recipe$compress$rank, stages, n_features
  )

  compress <- vector("list", length(stages))
  names(compress) <- stages
  scores <- vector("list", length(stages))
  names(scores) <- stages

  for (i in seq_along(stages)) {
    st <- stages[[i]]
    compress[[st]] <- .fit_block_pca(
      block_list[[st]],
      spec = recipe$compress,
      rank = compress_ranks[[st]],
      stage_name = st
    )
    scores[[st]] <- compress[[st]]$scores
  }

  mapping <- vector("list", length(stages))
  names(mapping) <- stages
  innovations <- vector("list", length(stages))
  names(innovations) <- stages

  if (is.null(recipe$decomposition)) {
    innovations <- scores
    for (st in stages) {
      mapping[[st]] <- list(
        B_hat = matrix(0, 0, ncol(scores[[st]])),
        method = NA_character_,
        tolerance = NA_real_,
        lambda = NA_real_
      )
    }
  } else {
    prev_scores <- list()
    for (i in seq_along(stages)) {
      st <- stages[[i]]
      H <- .bind_score_blocks(prev_scores, n)
      map <- .fit_stage_mapping(
        H, scores[[st]],
        spec = recipe$decomposition
      )
      mapping[[st]] <- map
      innovations[[st]] <- .apply_stage_mapping(H, scores[[st]], map$B_hat)
      prev_scores[[st]] <- scores[[st]]
    }
  }

  post <- NULL
  output <- innovations
  if (!is.null(recipe$post_compress)) {
    post_nfeat <- vapply(innovations, ncol, integer(1))
    post_ranks <- .resolve_stage_ranks(
      recipe$post_compress$rank, stages, post_nfeat
    )
    post <- vector("list", length(stages))
    names(post) <- stages
    for (st in stages) {
      post[[st]] <- .fit_block_pca(
        innovations[[st]],
        spec = recipe$post_compress,
        rank = post_ranks[[st]],
        stage_name = paste0(st, " (post-compress)")
      )
      output[[st]] <- post[[st]]$scores
    }
  }

  output_names <- vapply(
    seq_along(stages),
    function(i) .stage_output_name(stages[[i]], i, recipe),
    character(1)
  )
  names(output_names) <- stages
  training_output <- vector("list", length(stages))
  names(training_output) <- unname(output_names)
  for (i in seq_along(stages)) {
    st <- stages[[i]]
    out_nm <- output_names[[st]]
    mat <- output[[st]]
    colnames(mat) <- .component_colnames(out_nm, ncol(mat))
    training_output[[out_nm]] <- mat
  }

  structure(
    list(
      recipe = recipe,
      stage_order = stages,
      output_names = output_names,
      compress = compress,
      mapping = mapping,
      post_compress = post,
      training_scores = scores,
      training_innovations = innovations,
      residuals = training_output,
      n_samples = n,
      n_features_in = n_features,
      transform_version = HIERARCHY_TRANSFORM_VERSION,
      provenance = list(package = "imfeatures")
    ),
    class = "hierarchy_transform"
  )
}

#' Apply a fitted hierarchy transformation
#'
#' Row-independent apply path for a \code{\link{fit_hierarchy_transform}}
#' object. \code{apply_hierarchy_transform()} is the readable alias of
#' \code{predict()}.
#'
#' @param object A \code{hierarchy_transform}.
#' @param blocks,newdata New blocks with the same stages and columns as the
#'   training data. A named list or \code{feature_hierarchy}.
#' @param ... Unused.
#'
#' @return A named list of numeric matrices, one per output band.
#' @export
apply_hierarchy_transform <- function(object, blocks, ...) {
  predict(object, blocks, ...)
}

#' @rdname apply_hierarchy_transform
#' @export
#' @method predict hierarchy_transform
predict.hierarchy_transform <- function(object, newdata, ...) {
  if (!inherits(object, "hierarchy_transform")) {
    stop("Input 'object' must be of class 'hierarchy_transform'.")
  }
  if (missing(newdata)) {
    return(object$residuals)
  }
  .apply_hierarchy_transform(object, newdata)
}

#' @export
print.hierarchy_transform <- function(x, ...) {
  cat("Hierarchy transform")
  if (!is.null(x$recipe$decomposition)) {
    cat(" (ordered innovation, method=", x$recipe$decomposition$method, ")", sep = "")
  }
  cat("\n")
  cat("  Stages: ", paste(x$stage_order, collapse = ", "), "\n", sep = "")
  cat("  Output: ", paste(unname(x$output_names), collapse = ", "), "\n", sep = "")
  cr <- vapply(x$compress[x$stage_order], function(cf) cf$rank, integer(1))
  cat("  Compress ranks: ", paste(cr, collapse = ", "), "\n", sep = "")
  if (!is.null(x$post_compress)) {
    pr <- vapply(x$post_compress[x$stage_order], function(cf) cf$rank, integer(1))
    cat("  Post-compress ranks: ", paste(pr, collapse = ", "), "\n", sep = "")
  }
  cat("  Version: ", x$transform_version, "\n", sep = "")
  invisible(x)
}

#' @keywords internal
.apply_hierarchy_transform <- function(object, newdata) {
  block_list <- .as_block_list(newdata, arg = "newdata")
  stages <- object$stage_order
  missing <- setdiff(stages, names(block_list))
  if (length(missing)) {
    stop(
      "'newdata' must contain matrices for all stages present in the training object: ",
      paste(stages, collapse = ", ")
    )
  }
  ns <- vapply(block_list[stages], nrow, integer(1))
  if (length(unique(ns)) != 1L) {
    stop("All matrices in 'newdata' must have the same number of rows (samples).")
  }
  n <- ns[[1]]
  if (n == 0L) {
    stop("'newdata' matrices cannot have zero rows.")
  }

  scores <- vector("list", length(stages))
  names(scores) <- stages
  for (st in stages) {
    scores[[st]] <- .apply_block_pca(object$compress[[st]], block_list[[st]], st)
  }

  innovations <- scores
  if (!is.null(object$recipe$decomposition)) {
    prev_scores <- list()
    for (st in stages) {
      H <- .bind_score_blocks(prev_scores, n)
      innovations[[st]] <- .apply_stage_mapping(
        H, scores[[st]], object$mapping[[st]]$B_hat
      )
      prev_scores[[st]] <- scores[[st]]
    }
  }

  output <- innovations
  if (!is.null(object$post_compress)) {
    for (st in stages) {
      output[[st]] <- .apply_block_pca(
        object$post_compress[[st]],
        innovations[[st]],
        paste0(st, " (post-compress)")
      )
    }
  }

  out <- vector("list", length(stages))
  names(out) <- unname(object$output_names)
  for (st in stages) {
    out_nm <- object$output_names[[st]]
    mat <- output[[st]]
    colnames(mat) <- .component_colnames(out_nm, ncol(mat))
    out[[out_nm]] <- mat
  }
  out
}

#' @keywords internal
.fit_block_pca <- function(X, spec, rank, stage_name) {
  X <- as.matrix(X)
  storage.mode(X) <- "double"
  n <- nrow(X)
  p <- ncol(X)
  k_requested <- as.integer(rank)

  if (any(!is.finite(X))) {
    stop(sprintf("Stage '%s' contains non-finite values (NA/Inf).", stage_name))
  }
  if (n < 2L && k_requested > 0L) {
    stop(sprintf(
      "Stage '%s': Need >=2 samples for PCA; got N=%d for requested rank=%d",
      stage_name, n, k_requested
    ))
  }

  k <- min(k_requested, p, if (n > 0L) n - 1L else 0L)
  empty <- function() {
    list(
      center = if (p > 0L) rep(0, p) else numeric(0),
      scale = if (p > 0L) rep(1, p) else numeric(0),
      rotation = matrix(0, p, 0),
      singular_values = numeric(0),
      score_sds = numeric(0),
      scores = matrix(0, n, 0),
      scores_unweighted = matrix(0, n, 0),
      rank = 0L,
      colnames = colnames(X),
      feature_scale = spec$feature_scale,
      score_weighting = spec$score_weighting,
      method = spec$method,
      prcomp = NULL
    )
  }
  if (k <= 0L) {
    return(empty())
  }

  do_center <- spec$feature_scale %in% c("sd", "center")
  do_scale <- identical(spec$feature_scale, "sd")
  method <- spec$method
  if (identical(method, "eigencore")) {
    if (!requireNamespace("eigencore", quietly = TRUE)) {
      warning(
        "method = 'eigencore' was chosen, but 'eigencore' is not installed. Falling back to 'stats'.",
        call. = FALSE
      )
      method <- "stats"
    }
  }

  if (identical(method, "stats")) {
    pca <- stats::prcomp(
      X,
      center = do_center,
      scale. = do_scale,
      retx = TRUE,
      rank. = k
    )
  } else {
    pca <- .prcomp_eigencore(X, k = k, center = do_center, scale. = do_scale)
  }

  k <- ncol(pca$x)
  scores_unweighted <- pca$x[, seq_len(k), drop = FALSE]
  score_sds <- if (k > 0L) {
    apply(scores_unweighted, 2, stats::sd)
  } else {
    numeric(0)
  }
  score_sds[!is.finite(score_sds) | score_sds < .Machine$double.eps] <- 1

  scores <- if (identical(spec$score_weighting, "whitened") && k > 0L) {
    .plain_matrix(scale(scores_unweighted, center = FALSE, scale = score_sds + 1e-8))
  } else {
    .plain_matrix(scores_unweighted)
  }

  center <- if (!is.null(pca$center)) pca$center else FALSE
  scale_v <- if (!is.null(pca$scale)) pca$scale else FALSE
  rotation <- pca$rotation[, seq_len(k), drop = FALSE]
  singular_values <- if (!is.null(pca$sdev)) {
    pca$sdev[seq_len(k)]
  } else {
    apply(scores_unweighted, 2, stats::sd)
  }

  list(
    center = center,
    scale = scale_v,
    rotation = rotation,
    singular_values = singular_values,
    score_sds = unname(as.numeric(score_sds)),
    scores = scores,
    scores_unweighted = scores_unweighted,
    rank = as.integer(k),
    colnames = colnames(X),
    feature_scale = spec$feature_scale,
    score_weighting = spec$score_weighting,
    method = method,
    prcomp = pca
  )
}

#' @noRd
.prcomp_eigencore <- function(X, k, center = TRUE, scale. = FALSE) {
  n <- nrow(X)
  center_v <- if (isTRUE(center)) {
    unname(colMeans(X))
  } else {
    FALSE
  }
  if (isTRUE(scale.)) {
    scale_v <- apply(X, 2, stats::sd)
    scale_v[!is.finite(scale_v) | scale_v < .Machine$double.eps] <- 1
    scale_v <- unname(as.numeric(scale_v))
  } else {
    scale_v <- FALSE
  }
  Xs <- scale(X, center = center_v, scale = scale_v)
  fit <- eigencore::svd_partial(Xs, rank = k, vectors = "both")
  d <- eigencore::values(fit)
  rotation <- eigencore::right_vectors(fit)
  scores <- Xs %*% rotation
  sdev <- as.numeric(d) / sqrt(max(1, n - 1))
  structure(
    list(
      sdev = sdev,
      rotation = rotation,
      center = center_v,
      scale = scale_v,
      x = scores,
      certificate = eigencore::certificate(fit)
    ),
    class = "prcomp"
  )
}

#' @keywords internal
.apply_block_pca <- function(pca_fit, X, stage_name) {
  X <- as.matrix(X)
  storage.mode(X) <- "double"
  k <- pca_fit$rank
  n <- nrow(X)
  if (k == 0L) {
    return(matrix(0, n, 0))
  }

  train_names <- pca_fit$colnames
  new_names <- colnames(X)
  if (!is.null(train_names) && !is.null(new_names)) {
    if (!identical(new_names, train_names)) {
      if (setequal(new_names, train_names)) {
        X <- X[, train_names, drop = FALSE]
      } else {
        stop(sprintf(
          "Column names for stage '%s' in 'newdata' do not match or are not a permutation of training data column names.",
          stage_name
        ))
      }
    }
  } else if (!is.null(train_names) && is.null(new_names)) {
    stop(sprintf(
      "Stage '%s' training data had column names, but newdata does not. Cannot ensure correct column order.",
      stage_name
    ))
  } else if (is.null(train_names) && !is.null(new_names)) {
    warning(sprintf(
      "Stage '%s' training data had no column names, but newdata does. Proceeding by column index.",
      stage_name
    ), call. = FALSE)
  }

  expected_p <- nrow(pca_fit$rotation)
  if (ncol(X) != expected_p) {
    stop(sprintf(
      "Column count mismatch for stage '%s' in 'newdata'. Expected %d, got %d.",
      stage_name, expected_p, ncol(X)
    ))
  }

  Xs <- scale(X, center = pca_fit$center, scale = pca_fit$scale)
  Z <- Xs %*% pca_fit$rotation
  if (identical(pca_fit$score_weighting, "whitened")) {
    Z <- scale(Z, center = FALSE, scale = pca_fit$score_sds + 1e-8)
  }
  .plain_matrix(Z)
}

#' @keywords internal
.plain_matrix <- function(x) {
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  attr(x, "scaled:center") <- NULL
  attr(x, "scaled:scale") <- NULL
  x
}

#' @keywords internal
.bind_score_blocks <- function(score_list, n) {
  if (length(score_list) == 0L) {
    return(matrix(0, n, 0))
  }
  nonempty <- score_list[vapply(score_list, ncol, integer(1)) > 0L]
  if (length(nonempty) == 0L) {
    return(matrix(0, n, 0))
  }
  do.call(cbind, unname(nonempty))
}

#' @keywords internal
.fit_stage_mapping <- function(H, Z, spec) {
  p_h <- ncol(H)
  p_z <- ncol(Z)
  empty_B <- matrix(0, p_h, p_z)
  if (p_h == 0L || p_z == 0L) {
    return(list(
      B_hat = empty_B,
      method = spec$method,
      tolerance = spec$tolerance,
      lambda = spec$lambda
    ))
  }

  if (identical(spec$method, "ridge") &&
    is.numeric(spec$lambda) && spec$lambda > 0) {
    gram <- crossprod(H)
    diag(gram) <- diag(gram) + spec$lambda
    B <- solve(gram, crossprod(H, Z))
    return(list(
      B_hat = B,
      method = spec$method,
      tolerance = spec$tolerance,
      lambda = spec$lambda
    ))
  }

  B <- .svd_coefficient_map(H, Z, spec$tolerance)
  list(
    B_hat = B,
    method = spec$method,
    tolerance = spec$tolerance,
    lambda = spec$lambda
  )
}

#' @keywords internal
.svd_coefficient_map <- function(H, Z, tolerance) {
  nlv <- min(nrow(H), ncol(H))
  if (nlv == 0L) {
    return(matrix(0, ncol(H), ncol(Z)))
  }
  sv <- svd(H, nu = nlv, nv = nlv)
  if (!is.finite(sv$d[1]) || sv$d[1] < .Machine$double.eps) {
    return(matrix(0, ncol(H), ncol(Z)))
  }
  abs_rank_tol <- tolerance * max(dim(H)) * sv$d[1]
  keep <- sv$d > abs_rank_tol & sv$d > .Machine$double.eps
  if (!any(keep)) {
    return(matrix(0, ncol(H), ncol(Z)))
  }
  U <- sv$u[, keep, drop = FALSE]
  V <- sv$v[, keep, drop = FALSE]
  dinv <- 1 / sv$d[keep]
  V %*% (dinv * crossprod(U, Z))
}

#' @keywords internal
.apply_stage_mapping <- function(H, Z, B_hat) {
  if (ncol(Z) == 0L) {
    return(Z)
  }
  if (is.null(B_hat) || ncol(H) == 0L || nrow(B_hat) == 0L) {
    return(Z)
  }
  Z - H %*% B_hat
}

#' @keywords internal
.unscale_features <- function(Xs, center, scale) {
  if (!is.logical(scale)) {
    Xs <- sweep(Xs, 2, scale, `*`)
  }
  if (!is.logical(center)) {
    Xs <- sweep(Xs, 2, center, `+`)
  }
  Xs
}

#' @keywords internal
.reconstruct_from_pca <- function(pca_fit, scores) {
  Z <- scores
  if (identical(pca_fit$score_weighting, "whitened") && pca_fit$rank > 0L) {
    Z <- sweep(Z, 2, pca_fit$score_sds + 1e-8, `*`)
  }
  if (pca_fit$rank == 0L) {
    p <- if (!is.logical(pca_fit$center)) length(pca_fit$center) else nrow(pca_fit$rotation)
    return(matrix(0, nrow(scores), p))
  }
  Xs <- Z %*% t(pca_fit$rotation)
  .unscale_features(Xs, pca_fit$center, pca_fit$scale)
}
