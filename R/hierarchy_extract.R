#' Extract a named feature hierarchy from images
#'
#' Runs one batched multi-output forward pass and applies a per-stage pooler.
#' Convolutional activations are converted to NHWC before pooling. The virtual
#' tap \code{classifier_logits} is computed from \code{fc2} and the ImageNet
#' classifier weights without applying softmax.
#'
#' Spatial pooling is set per stage in \code{spec}; see \code{\link{poolers}}
#' for the options. Each block has one row per image and one column per
#' pooled feature, named as described in
#' \code{\link{pooler_feature_names}}. Full-resolution maps are pooled batch
#' by batch and never held for all images at once, so memory use depends on
#' \code{batch_size} and on the pooled width, not on the number of images.
#'
#' @param impaths Character vector of image paths, or a directory containing
#'   \code{jpg}/\code{jpeg}/\code{png} files.
#' @param spec A \code{\link{vgg_hierarchy_spec}} or other
#'   \code{feature_hierarchy_spec}.
#' @param model Optional preloaded Keras model. If \code{NULL}, the spec
#'   architecture is loaded with ImageNet weights.
#' @param batch_size Images per forward pass.
#' @param layout Layout of 4-D backend outputs; \code{"nhwc"} for Keras.
#'
#' @return A \code{\link{as_feature_hierarchy}} object. Its \code{schema}
#'   records the taps, the poolers, and the layout that produced the blocks.
#'
#' @examples
#' \dontrun{
#' imgs <- list.files("stimuli", pattern = "\\.jpg$", full.names = TRUE)
#'
#' # Depth-adaptive DCT pooling at the end of each VGG-16 block
#' raw <- extract_feature_hierarchy(imgs, spec = vgg16_block_end_spec())
#' vapply(raw$blocks, ncol, integer(1))
#'
#' # The same taps, with a 3 x 3 grid on every convolutional stage
#' taps <- vgg_block_end_taps("vgg16")
#' pooling <- lapply(taps, function(tap) adaptive_grid_pool(3, 3))
#' pooling[c("fc1", "logits")] <- list(identity_pool(), identity_pool())
#' spec <- vgg_hierarchy_spec("vgg16", taps = taps, pooling = pooling)
#' raw_grid <- extract_feature_hierarchy(imgs, spec = spec)
#' }
#' @seealso \code{\link{vgg_hierarchy_spec}}, \code{\link{poolers}}, and
#'   \code{vignette("spatial-pooling", package = "imfeatures")}.
#' @export
#' @importFrom keras3 application_vgg16 application_vgg19
extract_feature_hierarchy <- function(impaths,
                                      spec,
                                      model = NULL,
                                      batch_size = 16L,
                                      layout = c("nhwc", "nchw")) {
  if (!inherits(spec, "feature_hierarchy_spec")) {
    stop("'spec' must be created by vgg_hierarchy_spec() or a related helper.")
  }
  layout <- match.arg(layout)
  checkmate::assert_count(batch_size, positive = TRUE)
  batch_size <- as.integer(batch_size)
  impaths <- .resolve_image_paths(impaths)

  if (is.null(model)) {
    model <- .vgg_load_model(spec$architecture)
  }

  resolved <- .resolve_vgg_taps(spec$taps)
  logits_weights <- NULL
  if (length(resolved$logits_stages)) {
    logits_weights <- .vgg_logits_weights(model)
  }

  multi <- .vgg_multi_output_model(model, resolved$unique_layers)
  n <- length(impaths)
  stage_names <- names(spec$taps)
  collected <- stats::setNames(vector("list", length(stage_names)), stage_names)
  activation_dims <- stats::setNames(vector("list", length(stage_names)), stage_names)
  starts <- seq(1L, n, by = batch_size)

  for (s0 in starts) {
    idx <- s0:min(s0 + batch_size - 1L, n)
    preds <- .vgg_forward_batch(multi, impaths[idx], spec$target_size)
    .validate_vgg_batch_outputs(preds, length(resolved$unique_layers), length(idx))
    names(preds) <- resolved$unique_layers
    batch_blocks <- .pool_hierarchy_batch(
      preds, spec, resolved, logits_weights, layout
    )
    for (st in stage_names) {
      # rbind() drops attributes, so keep the activation shape separately.
      if (is.null(activation_dims[[st]])) {
        activation_dims[[st]] <- attr(batch_blocks[[st]], "activation_dim")
      }
      collected[[st]] <- rbind(collected[[st]], batch_blocks[[st]])
    }
  }

  for (st in stage_names) {
    storage.mode(collected[[st]]) <- "double"
  }
  collected <- .name_extracted_blocks(collected, spec, activation_dims, layout)

  as_feature_hierarchy(
    collected,
    sample_id = impaths,
    stage_order = stage_names,
    schema = list(taps = spec$taps, pooling = spec$pooling, layout = layout),
    provenance = list(
      architecture = spec$architecture,
      backend = spec$backend,
      target_size = spec$target_size,
      batch_size = batch_size,
      image_paths = impaths
    )
  )
}

#' @keywords internal
.resolve_image_paths <- function(impaths) {
  if (length(impaths) == 1L && dir.exists(impaths)) {
    orig_dir <- impaths
    impaths <- list.files(
      orig_dir,
      pattern = "\\.(jpg|jpeg|png)$",
      full.names = TRUE,
      ignore.case = TRUE
    )
    if (length(impaths) == 0L) {
      stop("No image files (jpg, jpeg, png) found in directory: ", orig_dir)
    }
  }
  assert_image(impaths)
  impaths
}

#' @keywords internal
.pool_hierarchy_batch <- function(preds, spec, resolved, logits_weights, layout) {
  out <- list()
  for (st in names(spec$taps)) {
    layer <- resolved$keras_taps[[st]]
    a <- preds[[layer]]
    if (st %in% resolved$logits_stages) {
      a <- .dense_logits(a, logits_weights$kernel, logits_weights$bias)
    }
    pooler <- spec$pooling[[st]]
    mat <- pool_activations(pooler, a, layout = layout)
    attr(mat, "activation_dim") <- dim(a)
    out[[st]] <- mat
  }
  out
}

#' @keywords internal
.name_extracted_blocks <- function(blocks, spec, activation_dims = NULL, layout = "nhwc") {
  for (st in names(blocks)) {
    mat <- blocks[[st]]
    pooler <- spec$pooling[[st]]
    d <- activation_dims[[st]]
    n_modes <- switch(pooler$type,
      gap = 1L,
      gmp = 1L,
      grid = pooler$ny * pooler$nx,
      dct = pooler$ny * pooler$nx,
      radial = length(pooler$breaks) - 1L,
      1L
    )
    if (length(d) == 4L && identical(pooler$type, "identity")) {
      # Flattened 4-D map: name every position and channel.
      hwc <- if (identical(layout, "nchw")) d[c(3L, 4L, 2L)] else d[2:4]
      colnames(mat) <- pooler_feature_names(pooler, st, hwc[3], spatial = hwc[1:2])
    } else if (length(d) == 4L && ncol(mat) %% n_modes == 0L) {
      colnames(mat) <- pooler_feature_names(pooler, st, ncol(mat) / n_modes)
    } else {
      # Dense activations pass through every pooler unchanged.
      colnames(mat) <- pooler_feature_names(identity_pool(), st, ncol(mat))
    }
    blocks[[st]] <- mat
  }
  blocks
}
