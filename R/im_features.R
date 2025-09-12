#' Compute similarity matrix for a set of image using feature vectors from keras model
#'
#' @name compute_feature_similarity
#' @rdname compute_feature_similarity
#' @importFrom furrr future_map
#' @importFrom proxy simil
#' @importFrom coop tcosine
#' @param impaths paths to image files (vector of file paths)
#' @param layers the layer indices
#' @param model the Keras model
#' @param target_size the target image dimensions for appropriate for model
#' @param spatial_pooling A character string specifying the type of spatial processing to apply to 4D feature maps (see \code{extract_features} for details)
#' @param metric the similarity metric to use, default is 'cosine' (see \code{proxy} package for allowable metrics)
#' @param lowmem logical, if TRUE use memory-efficient computation (default: TRUE)
#' @param cache_size maximum cache size in bytes for memoization (default: 2048 * 2048^2)
#' @param subsamp_prop proportion of features to subsample (0 to 1, default: 1 for no subsampling)
#' @return A list of similarity matrices, one for each layer
#' @importFrom memoise memoise
#' @importFrom cachem cache_mem
#' @importFrom progress progress_bar
#' @examples
#' \dontrun{
#' # Create a vector of image paths
#' img_dir <- system.file("extdata", package = "imfeatures")
#' img_paths <- list.files(img_dir, pattern = "\\.jpg$", full.names = TRUE)
#' 
#' # Compute similarity matrix using features from specific layers
#' sim_matrix <- compute_feature_similarity(
#'   impaths = img_paths,
#'   layers = c(10, 15),  # Two VGG16 layers
#'   model = NULL,  # Use default VGG16
#'   target_size = c(224, 224),
#'   metric = "cosine"
#' )
#' 
#' # Access similarity matrices for each layer
#' layer10_sim <- sim_matrix$layer_10
#' layer15_sim <- sim_matrix$layer_15
#' 
#' # Compute similarity with spatial pooling for efficiency
#' sim_pooled <- compute_feature_similarity(
#'   impaths = img_paths,
#'   layers = c(12),
#'   spatial_pooling = "avg",  # Average pool spatial dimensions
#'   metric = "cosine",
#'   lowmem = TRUE  # Memory-efficient computation
#' )
#' 
#' # Use subsampling for very large feature vectors
#' sim_subsampled <- compute_feature_similarity(
#'   impaths = img_paths,
#'   layers = c(10),
#'   subsamp_prop = 0.5,  # Use 50% of features
#'   metric = "euclidean"
#' )
#' 
#' # Visualize similarity matrix
#' heatmap(sim_matrix$layer_10, symm = TRUE)
#' }
#' @export
compute_feature_similarity <- function(impaths, layers, model=NULL, target_size=c(224,224),
                           spatial_pooling = "none",
                           metric="cosine", lowmem=TRUE,cache_size=2048 * 2048^2,
                           subsamp_prop=1) {

  assert_image(impaths)
  if (length(impaths) <= 1) {
    stop("need at least two images to compare")
  }
  checkmate::assert_number(subsamp_prop, lower = 0, upper = 1, finite = TRUE)

  if (is.null(model)) {
    model <- application_vgg16(weights = 'imagenet', include_top = TRUE)
  }

  out <- lapply(seq_along(layers), function(l) {
    m <- matrix(0, length(impaths), length(impaths))
    row.names(m) <- basename(impaths)
    colnames(m) <- basename(impaths)
    m
  })

  #imfeat <- memoise::memoise(extract_features, omit_args=c("model"), cache=cachem::cache_mem(max_size = 2044 * 2048^2))
  imfeat <- memoise::memoise(extract_features, cache=cachem::cache_mem(max_size = cache_size))

  pb <- progress_bar$new(total = length(impaths))


  if (lowmem) {
    for (i in 1:length(impaths)) {
      pb$tick()
      for (j in 1:length(impaths)) {
        if (i < j & i != j) {
          #print(j)
          fi <- imfeat(impaths[i], layers=layers, model=model,
                        spatial_pooling = spatial_pooling)
          fj <- imfeat(impaths[j], layers=layers, model=model,
                        spatial_pooling = spatial_pooling)
          for (k in 1:length(layers)) {
            m <- proxy::simil(as.vector(fi[[k]]), as.vector(fj[[k]]), method=metric, by_rows=FALSE)
            out[[k]][i,j] <- m[1,1]
          }
        }
      }
    }

    out <- lapply(out, function(m) {
      m[lower.tri(m)] <- t(m)[lower.tri(m)]
      m
    })

  } else{
    if (subsamp_prop < 1) {
      f1 <- extract_features(impaths[1], layers=layers, model=model,
                         spatial_pooling = spatial_pooling)$feature
      subsamp_ind <- lapply(f1, function(feat) {
        size <- max(1L, round(length(feat) * subsamp_prop))
        sample(seq_along(feat), size)
      })
    }

    featlist <- furrr::future_map(impaths, function(im) {
      feats <- extract_features(im, layers=layers, model=model,
                           spatial_pooling = spatial_pooling)$feature
      if (subsamp_prop < 1) {
        feats <- lapply(seq_along(feats), function(i) {
          feats[[i]][subsamp_ind[[i]]]
        })
      }
      feats
    })

    out <-  furrr::future_map(seq_along(layers), function(i) {
      mat <- do.call(rbind, lapply(featlist, function(x) as.vector(x[[i]])))

      if (metric == "cosine") {
          coop::tcosine(mat)
      } else {
         as.matrix(proxy::simil(mat, metric))
       }
     })

  }


  onames <- paste0("layer_", layers)
  names(out) <- onames

  out
}

#' @rdname compute_feature_similarity
#' @export
im_feature_sim <- compute_feature_similarity

.vgg16 <- NULL

vgg16 <- function() {
  if (is.null(.vgg16)) {
    .vgg16 <<- keras::application_vgg16(weights = 'imagenet', include_top = TRUE)
    .vgg16
  } else {
    .vgg16
  }
}

#' extract features from intermediate layers
#'
#' @param impath path to image file
#' @param layers the layer indices
#' @param model the Keras model
#' @param target_size the target image dimensions for approproate for model
#' @param spatial_pooling A character string specifying the type of spatial processing to apply to 4D feature maps (typically from convolutional layers).
#'        Options are:
#'        \itemize{
#'          \item{\code{"none"}: (Default) No spatial processing is applied; the full feature maps are returned (usually as a 4D array: 1 x H x W x C).}
#'          \item{\code{"avg"}: Global average pooling is applied across spatial dimensions (H, W), resulting in one value per channel (vector of length C).}
#'          \item{\code{"max"}: Global max pooling is applied across spatial dimensions (H, W), resulting in one value per channel (vector of length C).}
#'          \item{\code{"resize_HxW"}: Downsamples the spatial dimensions to \code{H} by \code{W} using bilinear interpolation, then flattens. Any value matching \code{"^resize_[0-9]+x[0-9]+$"} is accepted (e.g., \code{"resize_3x3"}, \code{"resize_7x7"}). Results in a vector of length \code{H * W * C}.}
#'        }
#'        This parameter only affects 4D outputs. For other layer types (e.g., 2D outputs like N x Features from dense layers, or already pooled features),
#'        this parameter is ignored, and features are returned as is. The handling of these raw features (e.g. flattening) is typically managed by downstream functions.
#' @importFrom keras application_vgg16 image_to_array imagenet_preprocess_input keras_model get_layer
#' @return A tibble with columns \code{image}, \code{layer} and a list-column \code{feature}.
#'   The tibble inherits class \code{imfeatures_feature_tbl} for dplyr compatibility.
#' @name extract_features
#' @rdname extract_features
#' @examples
#' \dontrun{
#' # Extract features from a single image using default VGG16 model
#' img_path <- system.file("extdata", "example.jpg", package = "imfeatures")
#' 
#' # Extract features from multiple layers
#' features <- extract_features(
#'   impath = img_path,
#'   layers = c(3, 5, 7),  # conv1_2, conv2_1, conv2_2
#'   model = NULL,  # Uses default VGG16
#'   target_size = c(224, 224)
#' )
#' 
#' # Extract features with spatial pooling
#' features_pooled <- extract_features(
#'   impath = img_path,
#'   layers = c(10, 12),  # Later convolutional layers
#'   spatial_pooling = "avg"  # Global average pooling
#' )
#' 
#' # Extract features with spatial resizing
#' features_resized <- extract_features(
#'   impath = img_path,
#'   layers = c(10),
#'   spatial_pooling = "resize_7x7"  # Resize spatial dimensions to 7x7
#' )
#' 
#' # Access the extracted features
#' layer3_features <- features$feature[[1]]  # Features from layer 3
#' dim(layer3_features)  # Check dimensions
#' }
#' @export
extract_features <- function(impath, layers, model=NULL, target_size=c(224,224),
                        spatial_pooling = "none") {

  assert_image(impath)
  checkmate::assert_vector(layers, min.len = 1)
  checkmate::assert_integerish(target_size, len = 2)

  # Validate spatial pooling argument. Accept 'none', 'avg', 'max' or
  # patterns of the form 'resize_HxW'
  valid_opts <- c("none", "avg", "max")
  if (!(spatial_pooling %in% valid_opts ||
        grepl("^resize_[0-9]+x[0-9]+$", spatial_pooling))) {
    stop("'spatial_pooling' must be 'none', 'avg', 'max', or 'resize_HxW'")
  }

  if (is.null(model)) {
    model <- application_vgg16(weights = 'imagenet', include_top = TRUE)
  }

  img <- .image_load_compat(impath, target_size = target_size)

  x <- image_to_array(img)

  ## iif this fails, it means 'numpy' not available...
  x <- array_reshape(x, c(1, dim(x)))
  x <- imagenet_preprocess_input(x)

  #subsamp_indices <- vector(length(layers), mode="list")

  features <- lapply(layers, function(layer) {
    lyr <- if (is.numeric(layer)) {
      get_layer(model, index = as.integer(layer))
    } else {
      get_layer(model, name = layer)
    }
    intermediate_layer_model <- keras_model(inputs = model$input,
                                            outputs = lyr$output)

    p <- predict(intermediate_layer_model, x)

    p <- .process_feature_map(p, spatial_pooling)
  })

  tbl <- tibble::tibble(
    image = rep(impath, length(layers)),
    layer = layers,
    feature = features
  )
new_feature_tbl(tbl)
}

#' @rdname extract_features
#' @export
im_features <- extract_features

#' @keywords internal
.process_feature_map <- function(p, spatial_pooling) {
  # Applies global pooling or resizing to a 4D feature tensor (1 x H x W x C) or returns input unchanged.
  if (!is.null(dim(p)) && length(dim(p)) == 4) {
    if (spatial_pooling == "avg") {
      return(as.vector(apply(p, MARGIN = c(1, 4), FUN = mean)))
    } else if (spatial_pooling == "max") {
      return(as.vector(apply(p, MARGIN = c(1, 4), FUN = max)))
    } else if (startsWith(spatial_pooling, "resize_")) {
      # Delegate to TensorFlow for resizing
      if (!requireNamespace("tensorflow", quietly = TRUE)) {
        warning("TensorFlow not available. Original features returned.")
        return(p)
      }
      tf <- reticulate::import("tensorflow", delay_load = TRUE)
      dims_str <- sub("resize_", "", spatial_pooling)
      target_dims_int <- tryCatch({ as.integer(strsplit(dims_str, "x")[[1]]) }, error = function(e) NULL)
      if (!is.null(target_dims_int) && length(target_dims_int) == 2 && !any(is.na(target_dims_int)) && all(target_dims_int > 0)) {
        p_tf <- tf$constant(p, dtype = tf$float32)
        p_resized_tf <- tf$image$resize(
          images = p_tf,
          size = list(as.integer(target_dims_int[1]), as.integer(target_dims_int[2])),
          method = tf$image$ResizeMethod$BILINEAR
        )
        return(as.vector(as.array(p_resized_tf)))
      } else {
        warning(sprintf("Invalid resize format or dimensions in: %s. Original features returned.", spatial_pooling))
        return(p)
      }
    } else if (spatial_pooling == "none") {
      return(p)
    }
  }
  # Non-4D inputs are returned unchanged
  return(p)
}

#' predict the class of an image using Keras model
#'
#' @inheritParams im_features
#' @param topn number of top predictions to return (default: 12)
#' @examples
#' \dontrun{
#' # Predict class of a single image
#' img_path <- system.file("extdata", "dog.jpg", package = "imfeatures")
#' 
#' # Use default VGG16 model trained on ImageNet
#' predictions <- im_predict(img_path, topn = 5)
#' print(predictions)  # Top 5 predicted classes with scores
#' 
#' # Use a custom pre-loaded model
#' library(keras)
#' resnet_model <- application_resnet50(weights = 'imagenet')
#' predictions_resnet <- im_predict(
#'   impath = img_path,
#'   model = resnet_model,
#'   target_size = c(224, 224),
#'   topn = 10
#' )
#' 
#' # Predict using VGG16-Places365 for scene recognition
#' places_model <- load_vgg16_places()
#' scene_predictions <- im_predict(
#'   impath = "path/to/landscape.jpg",
#'   model = places_model,
#'   topn = 3
#' )
#' # Will return scene categories like "mountain", "forest", etc.
#' }
#' @export
#' @importFrom dplyr top_n arrange desc
#' @importFrom keras imagenet_decode_predictions
im_predict <- function(impath, model=NULL, target_size=c(224,224), topn=12) {
  assert_image(impath)
  checkmate::assert_integerish(target_size, len = 2)
  assert_scalar(topn, "integer")
  if (is.null(model)) {
    model <- application_vgg16(weights = 'imagenet', include_top = TRUE)
  }

  img <- .image_load_compat(impath, target_size = target_size)
  x <- image_to_array(img)
  #x <- array_reshape(x, c(1, unlist(x$shape)))
  x <- array_reshape(x, c(1, dim(x)))
  x <- imagenet_preprocess_input(x)

  preds <- model %>% predict(x)

  if (model$name == "vgg16-places365") {
    data("places_cat365")
    data.frame(class_name=places_cat365$category, score=preds[1,]) %>% arrange(desc(score)) %>% top_n(topn)
  } else {
    imagenet_decode_predictions(preds,topn)
  }
}

#' @keywords internal
#' Image loader compatible with Keras 2/3 API changes
#'
#' Attempts to load images using keras.utils.load_img (Keras 2/3) or
#' keras.preprocessing.image.load_img (older), without passing deprecated
#' grayscale argument. Falls back to PIL if Keras is unavailable.
.image_load_compat <- function(path, target_size = NULL, color_mode = "rgb") {
  # Try Python Keras first
  k <- NULL
  if (reticulate::py_module_available("keras")) {
    k <- reticulate::import("keras", delay_load = TRUE)
  } else if (reticulate::py_module_available("tensorflow.keras")) {
    k <- reticulate::import("tensorflow.keras", delay_load = TRUE)
  }

  size_tuple <- NULL
  if (!is.null(target_size)) {
    if (length(target_size) != 2) stop("target_size must be length-2: c(height, width)")
    size_tuple <- reticulate::tuple(as.integer(target_size))
  }

  if (!is.null(k)) {
    # Prefer keras.utils.load_img if available
    if (reticulate::py_has_attr(k, "utils") && reticulate::py_has_attr(k$utils, "load_img")) {
      if (is.null(size_tuple)) return(k$utils$load_img(path, color_mode = color_mode))
      return(k$utils$load_img(path, color_mode = color_mode, target_size = size_tuple))
    }
    # Fallback to legacy preprocessing path
    if (reticulate::py_has_attr(k, "preprocessing") &&
        reticulate::py_has_attr(k$preprocessing, "image") &&
        reticulate::py_has_attr(k$preprocessing$image, "load_img")) {
      if (is.null(size_tuple)) return(k$preprocessing$image$load_img(path, color_mode = color_mode))
      return(k$preprocessing$image$load_img(path, color_mode = color_mode, target_size = size_tuple))
    }
  }

  # Final fallback: PIL
  if (reticulate::py_module_available("PIL.Image")) {
    PIL_img <- reticulate::import("PIL.Image", delay_load = TRUE)
    im <- PIL_img$open(path)
    im <- if (identical(color_mode, "rgb")) im$convert("RGB") else im
    if (!is.null(size_tuple)) {
      # PIL expects (width, height)
      im <- im$resize(reticulate::tuple(as.integer(rev(target_size))))
    }
    return(im)
  }

  stop("No suitable image loader available (keras/PIL not found).")
}

#p=reticulate::import("keras_models.models.pretrained.vgg16_places365")
#model=p$VGG16_Places365()
#target_size=c(224,224)
#intermediate_layer_model <- keras_model(inputs = model$input,
                                       # outputs = get_layer(model, index=index)$output)
#predict(intermediate_layer_model, x)
