#' Extract VGG-16 features by tier
#'
#' Convenience wrapper around \code{im_features()} to extract VGG-16 features grouped by spatial tiers:
#' \itemize{
#'   \item \code{"low"}: conv1_1, conv1_2, conv2_1, conv2_2
#'   \item \code{"mid"}: conv3_1 through conv4_3
#'   \item \code{"high"}: conv5_1 through conv5_3
#'   \item \code{"semantic"}: fc1 (fc6) and fc2 (fc7)
#' }
#' Layers are retrieved by name (e.g., \code{"block1_conv1"}) instead of numeric indices.
#'
#' @param impaths Character vector of image file paths.
#' @param tier Character; one of "low", "mid", "high", or "semantic".
#' @param model Preloaded Keras VGG-16 model object. If NULL, defaults to \code{keras3::application_vgg16(weights = 'imagenet')}.
#' @param target_size Numeric vector of length 2 specifying image resize dimensions (width, height).
#' @param pooling Character string specifying spatial pooling; passed to the \code{spatial_pooling} argument of \code{im_features}.
#'        Defaults to "avg" (global average pooling). Other options: "none", "max", "resize_3x3", "resize_5x5", "resize_7x7".
#' @param batch_size Integer; number of images per forward pass. Images are pushed
#'        through a single multi-output model in batches rather than one at a time,
#'        which is substantially faster for large image sets. Defaults to 8 to
#'        bound peak memory use for early VGG layers.
#' @return An S3 object of class \code{vgg_feature_set}, a list with components:
#' \describe{
#'   \item{features}{Numeric matrix (N_images × total_channels) of pooled features.}
#'   \item{image_paths}{Character vector of input image paths.}
#'   \item{tier}{The tier name.}
#'   \item{pooling}{Pooling type used.}
#'   \item{layer_indices}{Numeric indices of the selected layers (derived from \code{layer_names}).}
#'   \item{layer_names}{Character names of VGG-16 layers used.}
#'   \item{model_name}{Character, set to "vgg16".}
#'   \item{target_size}{Numeric vector of image resize dimensions.}
#'   \item{batch_size}{Integer number of images processed per forward pass.}
#' }
#' @export
#' @importFrom keras3 application_vgg16 keras_model get_layer image_to_array imagenet_preprocess_input
extract_vgg_features <- function(impaths,
                                 tier = c("low", "mid", "high", "semantic"),
                                 model = NULL,
                                 target_size = c(224, 224),
                                 pooling = "avg",
                                 batch_size = 8L) {
  assert_image(impaths)
  checkmate::assert_integerish(target_size, len = 2)
  assert_scalar(pooling, "character")
  checkmate::assert_count(batch_size, positive = TRUE)
  batch_size <- as.integer(batch_size)
  # Allow passing a directory containing images
  if (length(impaths) == 1 && dir.exists(impaths)) {
    orig_dir <- impaths
    impaths <- list.files(orig_dir,
      pattern = "\\.(jpg|jpeg|png)$",
      full.names = TRUE,
      ignore.case = TRUE
    )
    if (length(impaths) == 0) {
      stop("No image files (jpg, jpeg, png) found in directory: ", orig_dir)
    }
  }
  # Ensure all paths exist
  missing <- impaths[!file.exists(impaths)]
  if (length(missing) > 0) {
    stop("The following image files do not exist: ", paste(missing, collapse = ", "))
  }

  tier <- match.arg(tier)
  pooling <- match.arg(
    pooling,
    c("none", "avg", "max", "resize_3x3", "resize_5x5", "resize_7x7")
  )

  if (is.null(model)) {
    model <- keras3::application_vgg16(weights = "imagenet", include_top = TRUE)
  }

  # Define layer name map for VGG-16
  tier_map <- list(
    low = c("block1_conv1", "block1_conv2", "block2_conv1", "block2_conv2"),
    mid = c(
      "block3_conv1", "block3_conv2", "block3_conv3",
      "block4_conv1", "block4_conv2", "block4_conv3"
    ),
    high = c("block5_conv1", "block5_conv2", "block5_conv3"),
    semantic = c("fc1", "fc2")
  )
  layers <- tier_map[[tier]]

  # Get numeric indices for reference and store layer names
  all_names <- vapply(model$layers, function(l) l$name, character(1))
  layer_indices <- match(layers, all_names)
  layer_names <- layers

  # One model with every requested layer as an output, evaluated on batches.
  # Building a fresh keras_model per layer per image (and predicting on a batch
  # of 1) is what im_features() does; for many images that cost is dominated by
  # graph construction and tf.function retracing rather than by the convolutions.
  multi <- .vgg_multi_output_model(model, layers)

  rows <- vector("list", length(impaths))
  starts <- seq(1L, length(impaths), by = batch_size)
  for (s0 in starts) {
    idx <- s0:min(s0 + batch_size - 1L, length(impaths))
    preds <- .vgg_forward_batch(multi, impaths[idx], target_size)
    .validate_vgg_batch_outputs(preds, length(layers), length(idx))

    # Pool per image so that every spatial_pooling mode -- including "none" and
    # the resize_HxW variants, whose output layout is per-image -- keeps exactly
    # the semantics it had when images were processed one at a time.
    for (i in seq_along(idx)) {
      vecs <- lapply(seq_along(layers), function(j) {
        p <- preds[[j]]
        p_i <- if (length(dim(p)) == 4L) {
          p[i, , , , drop = FALSE]
        } else {
          p[i, , drop = FALSE]
        }
        .process_feature_map(p_i, pooling)
      })
      rows[[idx[i]]] <- unlist(vecs, use.names = FALSE)
    }
  }

  features <- do.call(rbind, rows)
  storage.mode(features) <- "double"

  res <- list(
    features = features,
    image_paths = impaths,
    tier = tier,
    pooling = pooling,
    layer_indices = layer_indices,
    layer_names = layer_names,
    model_name = "vgg16",
    target_size = target_size,
    batch_size = batch_size
  )
  class(res) <- "vgg_feature_set"
  res
}

#' Print method for vgg_feature_set objects
#'
#' Displays a summary of a VGG-16 feature set, including the tier,
#' number of images, feature dimensionality and pooling type.
#'
#' @param x A \code{vgg_feature_set} object.
#' @param ... Additional arguments (ignored).
#'
#' @examples
#' \dontrun{
#' img <- system.file("extdata", "cat.jpg", package = "imfeatures")
#' fs <- extract_vgg_features(img)
#' print(fs)
#' }
#' @export
print.vgg_feature_set <- function(x, ...) {
  cat("VGG-16 feature set\n")
  cat("  Tier:         ", x$tier, "\n")
  cat("  Images:       ", length(x$image_paths), "\n")
  cat("  Total dims:   ", ncol(x$features), "\n")
  cat("  Layers:       ", paste(x$layer_names, collapse = ", "), "\n")
  cat("  Pooling:      ", x$pooling, "\n")
  invisible(x)
}


# Build one Keras model exposing every requested layer as an output.
# Split out so tests can substitute it without a real Keras model.
.vgg_multi_output_model <- function(model, layers) {
  outputs <- lapply(layers, function(nm) keras3::get_layer(model, name = nm)$output)
  keras3::keras_model(inputs = model$input, outputs = outputs)
}

# Load a batch of images and run one forward pass, returning a list with one
# array per output layer (first dimension indexes the image within the batch).
.vgg_forward_batch <- function(multi, paths, target_size) {
  arrs <- lapply(paths, function(path) {
    tryCatch(
      keras3::image_to_array(.image_load_compat(path, target_size = target_size)),
      error = function(e) {
        stop(sprintf("Error processing image '%s': %s", path, e$message), call. = FALSE)
      }
    )
  })
  x <- array(0, dim = c(length(arrs), dim(arrs[[1]])))
  for (i in seq_along(arrs)) x[i, , , ] <- arrs[[i]]
  x <- keras3::imagenet_preprocess_input(x)

  preds <- stats::predict(multi, x, verbose = 0)
  if (!is.list(preds)) preds <- list(preds)
  preds
}

.validate_vgg_batch_outputs <- function(preds, expected_layers, batch_n) {
  if (length(preds) != expected_layers) {
    stop(sprintf(
      "Expected %d VGG layer output(s), received %d",
      expected_layers, length(preds)
    ), call. = FALSE)
  }

  invalid_batch <- vapply(preds, function(p) {
    output_dims <- dim(p)
    is.null(output_dims) || length(output_dims) < 2L || output_dims[1] != batch_n
  }, logical(1))
  if (any(invalid_batch)) {
    stop("VGG output batch dimension does not match the input batch",
      call. = FALSE
    )
  }

  invisible(TRUE)
}
