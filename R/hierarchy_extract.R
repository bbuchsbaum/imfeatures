#' Extract a named feature hierarchy from images
#'
#' Runs one batched multi-output forward pass and applies a per-stage pooler.
#' Convolutional activations are converted to NHWC before pooling. The virtual
#' tap \code{classifier_logits} is computed from \code{fc2} and the ImageNet
#' classifier weights without applying softmax.
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
#' @return A \code{\link{as_feature_hierarchy}} object.
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
  collected <- setNames(vector("list", length(stage_names)), stage_names)
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
      collected[[st]] <- rbind(collected[[st]], batch_blocks[[st]])
    }
  }

  for (st in stage_names) {
    storage.mode(collected[[st]]) <- "double"
  }
  collected <- .name_extracted_blocks(collected, spec)

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
.name_extracted_blocks <- function(blocks, spec) {
  for (st in names(blocks)) {
    mat <- blocks[[st]]
    pooler <- spec$pooling[[st]]
    tap <- spec$taps[[st]]
    if (identical(pooler$type, "identity") && tap %in% .vgg_dense_taps) {
      colnames(mat) <- pooler_feature_names(pooler, st, ncol(mat))
    } else if (identical(pooler$type, "identity")) {
      # 4-D flatten: infer H, W from ncol / channels if possible
      colnames(mat) <- pooler_feature_names(pooler, st, ncol(mat))
    } else {
      n_modes <- switch(pooler$type,
        gap = 1L,
        gmp = 1L,
        grid = pooler$ny * pooler$nx,
        dct = pooler$ny * pooler$nx,
        radial = length(pooler$breaks) - 1L,
        1L
      )
      if (n_modes > 0L && ncol(mat) %% n_modes == 0L) {
        n_channels <- ncol(mat) / n_modes
        colnames(mat) <- pooler_feature_names(pooler, st, n_channels)
      } else {
        colnames(mat) <- pooler_feature_names(identity_pool(), st, ncol(mat))
      }
    }
    blocks[[st]] <- mat
  }
  blocks
}
