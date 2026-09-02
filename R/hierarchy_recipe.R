#' Specify per-stage PCA compression
#'
#' @param rank Integer scalar, named integer vector, or \code{NULL}. A scalar
#'   is recycled across stages. \code{NULL} keeps up to 50 components per
#'   stage (or fewer when a stage has fewer columns).
#' @param method Character; \code{"stats"} uses \code{stats::prcomp},
#'   \code{"eigencore"} uses \code{eigencore::svd_partial} when installed.
#' @param feature_scale How to standardize raw columns before PCA:
#'   \code{"sd"} centers and scales, \code{"center"} centers only,
#'   \code{"none"} leaves columns unchanged.
#' @param score_weighting \code{"singular"} keeps PCA scores as \(UD\);
#'   \code{"whitened"} scales each retained score to unit training SD.
#'
#' @return An object of class \code{block_pca}.
#' @export
block_pca <- function(rank = NULL,
                      method = c("stats", "eigencore"),
                      feature_scale = c("sd", "center", "none"),
                      score_weighting = c("singular", "whitened")) {
  method <- match.arg(method)
  feature_scale <- match.arg(feature_scale)
  score_weighting <- match.arg(score_weighting)
  if (!is.null(rank)) {
    checkmate::assert_numeric(rank, lower = 0, finite = TRUE, any.missing = FALSE)
  }
  structure(
    list(
      rank = rank,
      method = method,
      feature_scale = feature_scale,
      score_weighting = score_weighting
    ),
    class = c("block_pca", "hierarchy_compressor")
  )
}

#' Specify ordered innovation residualization
#'
#' Fits a mapping from earlier-stage scores to the current stage on training
#' observations and stores the coefficient matrix so new rows can be
#' residualized independently.
#'
#' @param method \code{"svd"} uses a rank-aware least-squares mapping;
#'   \code{"ridge"} uses \eqn{\|Z - HB\|_F^2 + \lambda\|B\|_F^2}.
#' @param tolerance SVD rank tolerance factor, used as
#'   \code{s_i > tolerance * max(dim(H)) * s[1]}.
#' @param lambda Ridge penalty. A non-negative number, or the string
#'   \code{"blocked_cv"} (not yet implemented). Ignored when
#'   \code{method = "svd"} except that \code{"blocked_cv"} is still rejected.
#'
#' @return An object of class \code{ordered_innovation}.
#' @export
ordered_innovation <- function(method = c("svd", "ridge"),
                               tolerance = 1e-7,
                               lambda = 0) {
  method <- match.arg(method)
  checkmate::assert_number(tolerance, lower = 0, finite = TRUE)
  if (identical(lambda, "blocked_cv")) {
    lambda <- "blocked_cv"
  } else {
    checkmate::assert_number(lambda, lower = 0, finite = TRUE)
  }
  structure(
    list(
      method = method,
      tolerance = tolerance,
      lambda = lambda
    ),
    class = c("ordered_innovation", "hierarchy_decomposition")
  )
}

#' Compile a hierarchy transformation recipe
#'
#' A recipe records stage order, per-stage compression, an optional
#' innovation decomposition, and an optional post-residual compression.
#'
#' @param order Character vector of stage names, in residualization order.
#' @param compress A \code{\link{block_pca}} specification applied to each
#'   raw stage before decomposition.
#' @param decomposition An \code{\link{ordered_innovation}} specification, or
#'   \code{NULL} to emit compressed scores only.
#' @param post_compress Optional \code{\link{block_pca}} applied to each
#'   innovation (or compressed score if \code{decomposition} is \code{NULL}).
#' @param name_prefix Optional prefix for output band names, for example
#'   \code{"vgg"} yields \code{vgg_b1}, \code{vgg_b2_innovation}, ...
#' @param innovation_suffix Logical; if \code{TRUE} (default), stages after
#'   the first are named \code{*_innovation} when a decomposition is present.
#'
#' @return An object of class \code{hierarchy_recipe}.
#'
#' @examples
#' hierarchy_recipe(
#'   order = c("b1", "b2"),
#'   compress = block_pca(rank = 8),
#'   decomposition = ordered_innovation(method = "svd")
#' )
#' @export
hierarchy_recipe <- function(order,
                             compress,
                             decomposition = NULL,
                             post_compress = NULL,
                             name_prefix = "",
                             innovation_suffix = TRUE) {
  checkmate::assert_character(order, min.len = 1, any.missing = FALSE, unique = TRUE)
  if (!inherits(compress, "block_pca")) {
    stop("'compress' must be created by block_pca().")
  }
  if (!is.null(decomposition) && !inherits(decomposition, "hierarchy_decomposition")) {
    stop("'decomposition' must be created by ordered_innovation() or NULL.")
  }
  if (!is.null(post_compress) && !inherits(post_compress, "block_pca")) {
    stop("'post_compress' must be created by block_pca() or NULL.")
  }
  assert_scalar(name_prefix, "character")
  assert_scalar(innovation_suffix, "logical")
  structure(
    list(
      order = unname(as.character(order)),
      compress = compress,
      decomposition = decomposition,
      post_compress = post_compress,
      name_prefix = name_prefix,
      innovation_suffix = innovation_suffix
    ),
    class = "hierarchy_recipe"
  )
}

#' @export
print.hierarchy_recipe <- function(x, ...) {
  cat("Hierarchy recipe\n")
  cat("  Order: ", paste(x$order, collapse = " -> "), "\n", sep = "")
  cat(
    "  Compress: block_pca(rank=", .format_rank(x$compress$rank),
    ", scale=", x$compress$feature_scale,
    ", weighting=", x$compress$score_weighting, ")\n",
    sep = ""
  )
  if (is.null(x$decomposition)) {
    cat("  Decomposition: none\n")
  } else {
    cat(
      "  Decomposition: ordered_innovation(method=", x$decomposition$method,
      ")\n",
      sep = ""
    )
  }
  if (is.null(x$post_compress)) {
    cat("  Post-compress: none\n")
  } else {
    cat(
      "  Post-compress: block_pca(rank=", .format_rank(x$post_compress$rank),
      ")\n",
      sep = ""
    )
  }
  invisible(x)
}

#' @keywords internal
.format_rank <- function(rank) {
  if (is.null(rank)) {
    return("NULL")
  }
  if (is.null(names(rank))) {
    return(paste(rank, collapse = ","))
  }
  paste(sprintf("%s=%s", names(rank), rank), collapse = ",")
}

#' @keywords internal
.resolve_stage_ranks <- function(rank, stage_names, n_features) {
  n_stages <- length(stage_names)
  if (is.null(rank)) {
    ranks <- pmin(50L, as.integer(n_features))
    names(ranks) <- stage_names
    return(ranks)
  }
  rank <- as.integer(rank)
  if (length(rank) == 1L && is.null(names(rank))) {
    ranks <- rep(rank, n_stages)
    names(ranks) <- stage_names
    return(ranks)
  }
  if (!is.null(names(rank))) {
    missing <- setdiff(stage_names, names(rank))
    if (length(missing)) {
      stop(
        "Named rank is missing stages: ",
        paste(missing, collapse = ", "),
        "."
      )
    }
    return(stats::setNames(as.integer(rank[stage_names]), stage_names))
  }
  if (length(rank) == n_stages) {
    names(rank) <- stage_names
    return(rank)
  }
  stop(
    "rank must be NULL, a scalar, a named vector, or length equal to the number of stages (",
    n_stages, ")."
  )
}
