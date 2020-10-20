
#' compute similarity matrix for a set of image using feature vectors from keras model
#'
#' @import furrr, proxy
#' @param metric the similarity metric to use, default is 'cosine' (see \code{proxy} package for allowable metrics)
#' @inheritParams im_features
#' @export
im_feature_sim <- function(impaths, layers, model=NULL, target_size=c(224,224), metric="cosine") {

  featlist <- furrr::future_map(impaths, function(im) im_features(im, layers=layers, model=model))

  simlist <-  furrr::future_map(seq_along(layers), function(i) {
    mat <- do.call(rbind, lapply(featlist, function(x) as.vector(x[[i]])))

    if (metric == "cosine") {
      coop::tcosine(mat)
    } else {
      as.matrix(proxy::simil(mat, metric))
    }
  })

  simlist
}


#' extract features from intermediate layers
#'
#' @param impath path to image file
#' @param layers the layer indices
#' @param model the Keras model
#' @param target_size the target image dimensions for approproate for model
#' @import keras
#' @export
im_features <- function(impath, layers, model=NULL, target_size=c(224,224)) {
  if (is.null(model)) {
    model <- application_vgg16(weights = 'imagenet', include_top = TRUE)
  }

  img <- image_load(impath, target_size = target_size)

  x <- image_to_array(img)
  x <- array_reshape(x, c(1, dim(x)))
  x <- imagenet_preprocess_input(x)

  features <- lapply(layers, function(index) {
    intermediate_layer_model <- keras_model(inputs = model$input,
                                            outputs = get_layer(model, index=index)$output)

    predict(intermediate_layer_model, x)
  })

  #preds <- model %>% predict(x,6)
  #preds <- imagenet_decode_predictions(preds,10)
  features
}

#' predict the class of an image using Keras model
#' @inheritParams im_features
#' @export
im_predict <- function(impath, model=NULL, target_size=c(224,224), topn=12) {
  if (is.null(model)) {
    model <- application_vgg16(weights = 'imagenet', include_top = TRUE)
  }

  img <- image_load(impath, target_size = target_size)
  x <- image_to_array(img)
  x <- array_reshape(x, c(1, dim(x)))
  x <- imagenet_preprocess_input(x)

  preds <- model %>% predict(x)
  imagenet_decode_predictions(preds,topn)

}
