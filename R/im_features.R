#' Compute similarity matrix for a set of image using feature vectors from keras model
#'
#' @import furrr proxy
#' @param metric the similarity metric to use, default is 'cosine' (see \code{proxy} package for allowable metrics)
#' @inheritParams im_features
#' @import memoise
#' @import progress
#' @export
im_feature_sim <- function(impaths, layers, model=NULL, target_size=c(224,224),
                           metric="cosine", lowmem=TRUE,cache_size=2048 * 2048^2, 
                           subsamp_prop=1) {

  if (!(all(file.exists(impaths)))) {
    stop("not all files exist, check image paths.")
  }

  assertthat::assert_that(length(impaths) > 1, msg="need at least two images to compare")
  assertthat::assert_that(subsamp_prop <= 1, msg="subsamp_prop must be less than or equal to 1")
  assertthat::assert_that(subsamp_prop > 0, msg="subsamp_prop must be greater than 0")

  if (is.null(model)) {
    model <- application_vgg16(weights = 'imagenet', include_top = TRUE)
  }

  out <- lapply(seq_along(layers), function(l) {
    m <- matrix(0, length(impaths), length(impaths))
    row.names(m) <- basename(impaths)
    colnames(m) <- basename(impaths)
    m
  })

  #imfeat <- memoise::memoise(im_features, omit_args=c("model"), cache=cachem::cache_mem(max_size = 2044 * 2048^2))
  imfeat <<- memoise::memoise(im_features, cache=cachem::cache_mem(max_size = cache_size))

  pb <- progress_bar$new(total = length(impaths))


  if (lowmem) {
    for (i in 1:length(impaths)) {
      pb$tick()
      for (j in 1:length(impaths)) {
        if (i < j & i != j) {
          #print(j)
          fi <- imfeat(impaths[i], layers=layers, model=model)
          fj <- imfeat(impaths[j], layers=layers, model=model)
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
      f1 <- im_features(impaths[1], layers=layers, model=model)
      subsamp_ind <- lapply(f1, function(feat) {
        size <- max(1L, round(length(feat) * subsamp_prop))
        sample(seq_along(feat), size)
      })
    }

    featlist <- furrr::future_map(impaths, function(im) {
      feats <- im_features(im, layers=layers, model=model)
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
#'          \item{\code{"resize_HxW"}: (e.g., \code{"resize_3x3"}, \code{"resize_7x7"}) Downsamples the spatial dimensions (H, W) to H_new x W_new using bilinear interpolation, then flattens. Results in a vector of length H_new * W_new * C.}
#'        }
#'        This parameter only affects 4D outputs. For other layer types (e.g., 2D outputs like N x Features from dense layers, or already pooled features),
#'        this parameter is ignored, and features are returned as is. The handling of these raw features (e.g. flattening) is typically managed by downstream functions.
#' @import keras
#' @export
im_features <- function(impath, layers, model=NULL, target_size=c(224,224),
                        spatial_pooling = "none") {

  # Define allowed pooling options - extend this list for more resize options
  allowed_pooling_options <- c("none", "avg", "max", "resize_3x3", "resize_5x5", "resize_7x7")
  spatial_pooling <- match.arg(spatial_pooling, allowed_pooling_options)

  if (is.null(model)) {
    model <- application_vgg16(weights = 'imagenet', include_top = TRUE)
  }

  if (!file.exists(impath)) {
    stop(sprintf("Image path does not exist: %s", impath))
  }

  img <- image_load(impath, target_size = target_size)

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

  #preds <- model %>% predict(x,6)
  #preds <- imagenet_decode_predictions(preds,10)
  features
}

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
#' @export
#' @importFrom dplyr top_n arrange
im_predict <- function(impath, model=NULL, target_size=c(224,224), topn=12) {
  if (is.null(model)) {
    model <- application_vgg16(weights = 'imagenet', include_top = TRUE)
  }

  img <- image_load(impath, target_size = target_size)
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

#p=reticulate::import("keras_models.models.pretrained.vgg16_places365")
#model=p$VGG16_Places365()
#target_size=c(224,224)
#intermediate_layer_model <- keras_model(inputs = model$input,
                                       # outputs = get_layer(model, index=index)$output)
#predict(intermediate_layer_model, x)
