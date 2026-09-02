#' Build a named feature-hierarchy object
#'
#' Wraps a named list of aligned feature matrices (one per stage) into an
#' object that records sample identifiers, stage order, and optional schema
#' or provenance metadata.
#'
#' @param blocks Named list of numeric matrices, each with the same number of
#'   rows (observations). A \code{feature_hierarchy} is returned unchanged.
#' @param sample_id Optional vector of sample identifiers of length
#'   \code{nrow}. If omitted, row names of the first block are used, otherwise
#'   \code{seq_len(nrow)}.
#' @param stage_order Optional character vector giving the intended stage
#'   order. Must be a permutation of \code{names(blocks)}. Defaults to the
#'   list order.
#' @param schema Optional list describing column-name conventions or pooling.
#' @param provenance Optional list of extraction metadata.
#'
#' @return An object of class \code{feature_hierarchy} with elements
#'   \code{blocks}, \code{sample_id}, \code{stage_order}, \code{schema}, and
#'   \code{provenance}.
#'
#' @examples
#' blocks <- list(
#'   b1 = matrix(rnorm(20), 10, 2),
#'   b2 = matrix(rnorm(30), 10, 3)
#' )
#' as_feature_hierarchy(blocks)
#' @export
as_feature_hierarchy <- function(blocks,
                                 sample_id = NULL,
                                 stage_order = NULL,
                                 schema = NULL,
                                 provenance = NULL) {
  if (inherits(blocks, "feature_hierarchy")) {
    return(blocks)
  }
  blocks <- .validate_block_list(blocks, arg = "blocks")
  n <- nrow(blocks[[1]])
  if (is.null(stage_order)) {
    stage_order <- names(blocks)
  } else {
    checkmate::assert_character(stage_order, any.missing = FALSE, unique = TRUE)
    if (!setequal(stage_order, names(blocks))) {
      stop("'stage_order' must be a permutation of the block names.")
    }
    blocks <- blocks[stage_order]
  }
  if (is.null(sample_id)) {
    sample_id <- rownames(blocks[[1]])
    if (is.null(sample_id)) {
      sample_id <- seq_len(n)
    }
  }
  if (length(sample_id) != n) {
    stop("'sample_id' must have length equal to the number of rows.")
  }
  structure(
    list(
      blocks = blocks,
      sample_id = sample_id,
      stage_order = unname(as.character(stage_order)),
      schema = schema,
      provenance = provenance
    ),
    class = "feature_hierarchy"
  )
}

#' Subset the observation rows of feature blocks
#'
#' @param blocks A named list of matrices or a \code{feature_hierarchy}.
#' @param idx Integer, numeric, or logical index of rows to keep.
#'
#' @return An object of the same type as \code{blocks}, with rows subset.
#' @export
subset_feature_blocks <- function(blocks, idx) {
  if (inherits(blocks, "feature_hierarchy")) {
    n <- nrow(blocks$blocks[[1]])
    idx <- .as_row_index(idx, n)
    blocks$blocks <- lapply(blocks$blocks, function(m) m[idx, , drop = FALSE])
    blocks$sample_id <- blocks$sample_id[idx]
    return(blocks)
  }
  checkmate::assert_list(blocks, names = "named", min.len = 1)
  n <- nrow(as.matrix(blocks[[1]]))
  idx <- .as_row_index(idx, n)
  lapply(blocks, function(m) {
    m <- as.matrix(m)
    m[idx, , drop = FALSE]
  })
}

#' @export
print.feature_hierarchy <- function(x, ...) {
  n <- nrow(x$blocks[[1]])
  cat("Feature hierarchy\n")
  cat("  Samples: ", n, "\n", sep = "")
  cat("  Stages:  ", paste(x$stage_order, collapse = ", "), "\n", sep = "")
  dims <- vapply(x$blocks[x$stage_order], ncol, integer(1))
  cat(
    "  Dims:    ",
    paste(sprintf("%s=%d", names(dims), dims), collapse = ", "),
    "\n",
    sep = ""
  )
  invisible(x)
}

#' @keywords internal
.validate_block_list <- function(blocks, arg = "blocks", label = "Stage") {
  checkmate::assert_list(blocks, names = "named", min.len = 1)
  if (anyDuplicated(names(blocks))) {
    stop(sprintf("'%s' must have unique names.", arg))
  }
  blocks <- lapply(blocks, function(x) {
    m <- as.matrix(x)
    storage.mode(m) <- "double"
    m
  })
  ns <- vapply(blocks, nrow, integer(1))
  if (length(unique(ns)) != 1L) {
    counts_str <- paste(sprintf("%s=%d", names(ns), ns), collapse = ", ")
    stop(sprintf(
      "All matrices in '%s' must have the same number of rows (samples); got: %s",
      arg, counts_str
    ))
  }
  if (ns[[1]] == 0L) {
    stop(sprintf("'%s' matrices cannot have zero rows.", arg))
  }
  for (nm in names(blocks)) {
    if (any(!is.finite(blocks[[nm]]))) {
      stop(sprintf("%s '%s' contains non-finite values (NA/Inf).", label, nm))
    }
  }
  blocks
}

#' @keywords internal
.as_block_list <- function(x, arg = "blocks") {
  if (inherits(x, "feature_hierarchy")) {
    return(x$blocks[x$stage_order])
  }
  .validate_block_list(x, arg = arg)
}

#' @keywords internal
.as_row_index <- function(idx, n) {
  if (is.logical(idx)) {
    if (length(idx) != n) {
      stop("'idx' logical index must have length ", n, ".")
    }
    return(idx)
  }
  checkmate::assert_integerish(idx, min.len = 1, any.missing = FALSE)
  idx <- as.integer(idx)
  if (any(idx < 1L | idx > n)) {
    stop("'idx' contains positions outside 1:", n, ".", sep = "")
  }
  idx
}

#' @keywords internal
.n_rows_blocks <- function(blocks) {
  if (inherits(blocks, "feature_hierarchy")) {
    return(nrow(blocks$blocks[[1]]))
  }
  nrow(as.matrix(blocks[[1]]))
}
