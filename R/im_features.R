
#' compute similarity matrix for a set of image using feature vectors from keras model
#'
#' @import furrr proxy
#' @param metric the similarity metric to use, default is 'cosine' (see \code{proxy} package for allowable metrics)
#' @inheritParams im_features
#' @import memoise
#' @import progress
#' @export
im_feature_sim <- function(impaths, layers, model=NULL, target_size=c(224,224),
                           metric="cosine", lowmem=TRUE) {

  if (!(all(file.exists(impaths)))) {
    stop("not all files fout, check image paths.")
  }
  if (is.null(model)) {
    model <- application_vgg16(weights = 'imagenet', include_top = TRUE)
  }

  out <- lapply(seq_along(layers), function(l) {
    m <- matrix(0, length(impaths), length(impaths))
    row.names(m) <- basename(impaths)
    colnames(m) <- basename(impaths)
    m
  })

  imfeat <- memoise::memoise(im_features)

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
    featlist <- furrr::future_map(impaths, function(im) im_features(im, layers=layers, model=model))

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

.vgg16 <<- NULL

vgg16 <- function() {
  if (is.null(.vgg16)) {
    .vgg16 <<- application_vgg16(weights = 'imagenet', include_top = TRUE)
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
#' @import keras
#' @export
im_features <- function(impath, layers, model=NULL, target_size=c(224,224)) {
  if (is.null(model)) {
    model <- application_vgg16(weights = 'imagenet', include_top = TRUE)
  }

  img <- image_load(impath, target_size = target_size)

  x <- image_to_array(img)
  x <- array_reshape(x, c(1, dim(x)))
  #x <- array_reshape(x, c(1, unlist(x$shape)))
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
#'
#' @inheritParams im_features
#' @export
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
  imagenet_decode_predictions(preds,topn)

}
