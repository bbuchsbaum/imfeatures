#' Extract VGG-16 features by tier
#'
#' Convenience wrapper around \code{im_features()} to extract VGG-16 features grouped by spatial tiers:
#' \\itemize{
#'   \\item{\\code{"low"}: conv1_1, conv1_2, conv2_1, conv2_2}
#'   \\item{\\code{"mid"}: conv3_1–conv4_3}
#'   \\item{\\code{"high"}: conv5_1–conv5_3}
#'   \\item{\\code{"semantic"}: fc1 (fc6) and fc2 (fc7)}
#' }
#'
#' @param impaths Character vector of image file paths.
#' @param tier Character; one of "low", "mid", "high", or "semantic".
#' @param model Preloaded Keras VGG-16 model object. If NULL, defaults to \code{keras::application_vgg16(weights = 'imagenet')}. 
#' @param target_size Numeric vector of length 2 specifying image resize dimensions (width, height). 
#' @param pooling Character string specifying spatial pooling; passed to the \code{spatial_pooling} argument of \code{im_features}. 
#'        Defaults to "avg" (global average pooling). Other options: "none", "max", "resize_3x3", "resize_5x5", "resize_7x7". 
#' @return An S3 object of class \code{vgg_feature_set}, a list with components:
#' \describe{
#'   \item{features}{Numeric matrix (N_images × total_channels) of pooled features.}
#'   \item{image_paths}{Character vector of input image paths.}
#'   \item{tier}{The tier name.}
#'   \item{pooling}{Pooling type used.}
#'   \item{layer_indices}{Numeric indices of VGG-16 layers used.}
#'   \item{layer_names}{Character names of VGG-16 layers.}
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

  # Define layer index map for VGG-16
  tier_map <- list(
    low = c(1L, 2L, 4L, 5L),           # conv1_1, conv1_2, conv2_1, conv2_2
    mid = c(7L,  8L,  9L, 11L, 12L, 13L), # conv3_1–conv4_3
    high = c(15L, 16L, 17L),            # conv5_1–conv5_3
    semantic = c(20L, 21L)              # fc1 (fc6), fc2 (fc7)
  )
  layers <- tier_map[[tier]]

  # Get layer names
  layer_names <- vapply(
    layers,
    function(idx) keras::get_layer(model, index = idx)$name,
    character(1)
  )

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
    layer_indices = layers,
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