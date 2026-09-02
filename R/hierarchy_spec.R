#' Specify a VGG feature hierarchy
#'
#' @param architecture \code{"vgg16"} or \code{"vgg19"}.
#' @param taps Named character vector mapping stage names to Keras layer
#'   names. The virtual tap \code{"classifier_logits"} requests pre-softmax
#'   ImageNet logits. Defaults to \code{\link{vgg_block_end_taps}}.
#' @param pooling Named list of \code{imfeatures_pooler} objects, one per
#'   stage. A single pooler is recycled across stages. Dense taps default to
#'   \code{\link{identity_pool}}; convolutional taps default to
#'   \code{\link{global_average_pool}}.
#' @param target_size Image resize dimensions passed to Keras.
#'
#' @return An object of class \code{c("vgg_hierarchy_spec", "feature_hierarchy_spec")}.
#' @export
vgg_hierarchy_spec <- function(architecture = c("vgg16", "vgg19"),
                               taps = NULL,
                               pooling = NULL,
                               target_size = c(224, 224)) {
  architecture <- match.arg(architecture)
  if (is.null(taps)) {
    taps <- vgg_block_end_taps(architecture)
  }
  checkmate::assert_character(taps, min.len = 1, any.missing = FALSE, names = "named")
  if (anyDuplicated(names(taps))) {
    stop("'taps' must have unique stage names.")
  }
  checkmate::assert_integerish(target_size, len = 2, lower = 1)
  pooling <- .normalize_spec_pooling(pooling, taps)
  structure(
    list(
      architecture = architecture,
      taps = taps,
      pooling = pooling,
      target_size = as.integer(target_size),
      backend = "keras"
    ),
    class = c("vgg_hierarchy_spec", "feature_hierarchy_spec")
  )
}

#' VGG-16 block-end hierarchy with depth-adaptive DCT pooling
#'
#' One tap at the end of each convolutional block, plus \code{fc1} and
#' pre-softmax logits. Spatial modes decrease from \(6\times 6\) in block 1
#' to \(2\times 2\) in block 5.
#'
#' @inheritParams vgg_hierarchy_spec
#' @export
vgg16_block_end_spec <- function(target_size = c(224, 224)) {
  taps <- vgg_block_end_taps("vgg16")
  vgg_hierarchy_spec(
    architecture = "vgg16",
    taps = taps,
    pooling = .vgg_depth_adaptive_pooling(),
    target_size = target_size
  )
}

#' VGG-19 block-end hierarchy with depth-adaptive DCT pooling
#'
#' @inheritParams vgg_hierarchy_spec
#' @export
vgg19_block_end_spec <- function(target_size = c(224, 224)) {
  taps <- vgg_block_end_taps("vgg19")
  vgg_hierarchy_spec(
    architecture = "vgg19",
    taps = taps,
    pooling = .vgg_depth_adaptive_pooling(),
    target_size = target_size
  )
}

#' @export
print.vgg_hierarchy_spec <- function(x, ...) {
  cat("VGG hierarchy spec (", x$architecture, ")\n", sep = "")
  cat("  Stages: ", paste(names(x$taps), collapse = ", "), "\n", sep = "")
  cat("  Taps:   ", paste(unname(x$taps), collapse = ", "), "\n", sep = "")
  invisible(x)
}

#' @keywords internal
.normalize_spec_pooling <- function(pooling, taps) {
  if (is.null(pooling)) {
    return(.vgg_default_pooling(taps))
  }
  if (inherits(pooling, "imfeatures_pooler")) {
    out <- rep(list(pooling), length(taps))
    names(out) <- names(taps)
    return(out)
  }
  checkmate::assert_list(pooling, min.len = 1)
  if (is.null(names(pooling))) {
    if (length(pooling) == 1L && inherits(pooling[[1]], "imfeatures_pooler")) {
      out <- rep(pooling[1], length(taps))
      names(out) <- names(taps)
      return(out)
    }
    stop("'pooling' must be a named list of poolers or a single pooler.")
  }
  missing <- setdiff(names(taps), names(pooling))
  if (length(missing)) {
    stop("Pooling is missing stages: ", paste(missing, collapse = ", "), ".")
  }
  out <- pooling[names(taps)]
  bad <- !vapply(out, inherits, logical(1), "imfeatures_pooler")
  if (any(bad)) {
    stop("Each pooling entry must be an imfeatures_pooler.")
  }
  out
}
