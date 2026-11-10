#' Specify a VGG feature hierarchy
#'
#' A spec names the layers to tap and says how each one is pooled over
#' space. Pass it to \code{\link{extract_feature_hierarchy}} to get one
#' feature block per stage.
#'
#' @param architecture \code{"vgg16"} or \code{"vgg19"}.
#' @param taps Named character vector mapping stage names to Keras layer
#'   names. The virtual tap \code{"classifier_logits"} requests pre-softmax
#'   ImageNet logits. Defaults to \code{\link{vgg_block_end_taps}}.
#' @param pooling How to pool each stage over space; see
#'   \code{\link{poolers}} for the options. Supply one of:
#'   \itemize{
#'     \item \code{NULL} (default): \code{\link{global_average_pool}} for
#'       convolutional taps and \code{\link{identity_pool}} for dense taps.
#'     \item A single pooler, used for every stage. Dense stages have no
#'       spatial axes and pass through unchanged whatever the pooler.
#'     \item A named list with one pooler per stage. Names must cover every
#'       stage in \code{taps}.
#'   }
#' @param target_size Image resize dimensions passed to Keras.
#'
#' @return An object of class \code{c("vgg_hierarchy_spec", "feature_hierarchy_spec")}.
#'
#' @examples
#' # Default: global average pooling on convolutional stages
#' vgg_hierarchy_spec("vgg16")
#'
#' # One pooler for every stage
#' vgg_hierarchy_spec("vgg16", pooling = adaptive_grid_pool(2, 2))
#'
#' # A different pooler per stage
#' vgg_hierarchy_spec(
#'   "vgg16",
#'   taps = c(b4 = "block4_conv3", b5 = "block5_conv3", fc1 = "fc1"),
#'   pooling = list(
#'     b4 = dct_pool(3, 3),
#'     b5 = radial_pool(),
#'     fc1 = identity_pool()
#'   )
#' )
#'
#' @seealso \code{\link{vgg16_block_end_spec}} for a ready-made spec,
#'   \code{\link{poolers}}, and
#'   \code{vignette("spatial-pooling", package = "imfeatures")}.
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
#' pre-softmax logits. Convolutional blocks use \code{\link{dct_pool}}, and
#' the number of spatial modes decreases from 6 x 6 in block 1 to 2 x 2 in
#' block 5. Early blocks have few channels and fine spatial structure, so
#' they keep more spatial modes; late blocks have many channels and coarse
#' maps, so they keep fewer. This keeps the stages similar in width.
#'
#' With the default 224 x 224 input the VGG-16 stages are:
#'
#' \tabular{lllrrr}{
#'   \strong{Stage} \tab \strong{Layer} \tab \strong{Pooler} \tab \strong{Map} \tab \strong{Channels} \tab \strong{Columns} \cr
#'   \code{b1} \tab \code{block1_conv2} \tab \code{dct_pool(6, 6)} \tab 224 x 224 \tab 64 \tab 2304 \cr
#'   \code{b2} \tab \code{block2_conv2} \tab \code{dct_pool(5, 5)} \tab 112 x 112 \tab 128 \tab 3200 \cr
#'   \code{b3} \tab \code{block3_conv3} \tab \code{dct_pool(4, 4)} \tab 56 x 56 \tab 256 \tab 4096 \cr
#'   \code{b4} \tab \code{block4_conv3} \tab \code{dct_pool(3, 3)} \tab 28 x 28 \tab 512 \tab 4608 \cr
#'   \code{b5} \tab \code{block5_conv3} \tab \code{dct_pool(2, 2)} \tab 14 x 14 \tab 512 \tab 2048 \cr
#'   \code{fc1} \tab \code{fc1} \tab \code{identity_pool()} \tab -- \tab 4096 \tab 4096 \cr
#'   \code{logits} \tab \code{classifier_logits} \tab \code{identity_pool()} \tab -- \tab 1000 \tab 1000
#' }
#'
#' The first DCT column of every channel is its global average, so each
#' convolutional block contains the usual average-pooled features plus
#' low-frequency layout. DCT columns are not on a common scale across modes
#' or stages (see \code{\link{dct_pool}}); standardize them before
#' combining, as \code{\link{block_pca}} does by default.
#'
#' \code{vgg19_block_end_spec()} uses the same poolers on the VGG-19
#' block-end layers (\code{block3_conv4}, \code{block4_conv4},
#' \code{block5_conv4}); map sizes and column counts are identical.
#'
#' @inheritParams vgg_hierarchy_spec
#' @return A \code{vgg_hierarchy_spec}; see \code{\link{vgg_hierarchy_spec}}.
#' @examples
#' spec <- vgg16_block_end_spec()
#' spec
#' spec$pooling$b1
#' @seealso \code{\link{vgg_hierarchy_spec}} to choose other taps or poolers.
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

#' @rdname vgg16_block_end_spec
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
  cat("  Pooling:\n")
  width <- max(nchar(names(x$taps)))
  for (st in names(x$taps)) {
    cat(sprintf("    %-*s  ", width, st))
    print(x$pooling[[st]])
  }
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
