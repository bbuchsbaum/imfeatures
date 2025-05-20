#' Extract VGG-16 features by tier
#'
#' Convenience wrapper around \code{im_features()} to extract VGG-16 features grouped by spatial tiers:
#' \\itemize{
#'   \\item{\\code{"low"}: conv1_1, conv1_2, conv2_1, conv2_2}
#'   \\item{\\code{"mid"}: conv3_1–conv4_3}
#'   \\item{\\code{"high"}: conv5_1–conv5_3}
#'   \\item{\\code{"semantic"}: fc1 (fc6) and fc2 (fc7)}
#' }
#' Layers are retrieved by name (e.g., \code{"block1_conv1"}) instead of numeric indices.
#'
#' @param impaths Character vector of image file paths.
#' @param tier Character; one of "low", "mid", "high", or "semantic".
#' @param model Preloaded Keras VGG-16 model object. If NULL, defaults to \code{keras::application_vgg16(pretrained = 'imagenet')}.
#' @param target_size Numeric vector of length 2 specifying image resize dimensions (width, height).
#' @param pooling Character string specifying spatial pooling; passed to \code{im_features::spatial_pooling}.
#'        Defaults to "avg" (global average pooling). Other options: "none", "max", "resize_3x3", "resize_5x5", "resize_7x7".
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
#' }
#' @export
#' @import keras
extract_vgg_features <- function(impaths,
                                 tier = c("low", "mid", "high", "semantic"),
                                 model = NULL,
                                 target_size = c(224, 224),
                                 pooling = "avg") {
  # Allow passing a directory containing images
  if (length(impaths) == 1 && dir.exists(impaths)) {
    orig_dir <- impaths
    impaths <- list.files(orig_dir,
                          pattern = "\\.(jpg|jpeg|png)$",
                          full.names = TRUE,
                          ignore.case = TRUE)
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
  pooling <- match.arg(pooling,
                       c("none", "avg", "max", "resize_3x3", "resize_5x5", "resize_7x7"))

  if (is.null(model)) {
    model <- keras::application_vgg16(weights = 'imagenet', include_top = TRUE)
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

  # Extract features for each image; wrap errors per image
  feats_list <- lapply(impaths, function(path) {
    tryCatch(
      im_features(
        impath = path,
        layers = layers,
        model = model,
        target_size = target_size,
        spatial_pooling = pooling
      ),
      error = function(e) {
        stop(sprintf("Error processing image '%s': %s", path, e$message), call. = FALSE)
      }
    )
  })

  # Combine into matrix: N_images x total_features
  features <- do.call(
    rbind,
    lapply(feats_list, function(x) unlist(x, use.names = FALSE))
  )

  res <- list(
    features = features,
    image_paths = impaths,
    tier = tier,
    pooling = pooling,
    layer_indices = layer_indices,
    layer_names = layer_names,
    model_name = "vgg16",
    target_size = target_size
  )
  class(res) <- "vgg_feature_set"
  res
}

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