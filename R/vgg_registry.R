#' Block-end tap names for a VGG architecture
#'
#' @param architecture \code{"vgg16"} or \code{"vgg19"}.
#' @return A named character vector of Keras layer names, including the
#'   virtual tap \code{classifier_logits}.
#' @export
vgg_block_end_taps <- function(architecture = c("vgg16", "vgg19")) {
  architecture <- match.arg(architecture)
  switch(architecture,
    vgg16 = c(
      b1 = "block1_conv2",
      b2 = "block2_conv2",
      b3 = "block3_conv3",
      b4 = "block4_conv3",
      b5 = "block5_conv3",
      fc1 = "fc1",
      logits = "classifier_logits"
    ),
    vgg19 = c(
      b1 = "block1_conv2",
      b2 = "block2_conv2",
      b3 = "block3_conv4",
      b4 = "block4_conv4",
      b5 = "block5_conv4",
      fc1 = "fc1",
      logits = "classifier_logits"
    )
  )
}

#' @keywords internal
.vgg_dense_taps <- c("fc1", "fc2", "classifier_logits", "predictions")

#' @keywords internal
.vgg_default_pooling <- function(taps) {
  out <- lapply(taps, function(tap) {
    if (tap %in% .vgg_dense_taps) identity_pool() else global_average_pool()
  })
  names(out) <- names(taps)
  out
}

#' @keywords internal
.vgg_depth_adaptive_pooling <- function() {
  list(
    b1 = dct_pool(6, 6),
    b2 = dct_pool(5, 5),
    b3 = dct_pool(4, 4),
    b4 = dct_pool(3, 3),
    b5 = dct_pool(2, 2),
    fc1 = identity_pool(),
    logits = identity_pool()
  )
}

#' @keywords internal
.vgg_load_model <- function(architecture) {
  switch(architecture,
    vgg16 = keras3::application_vgg16(weights = "imagenet", include_top = TRUE),
    vgg19 = keras3::application_vgg19(weights = "imagenet", include_top = TRUE),
    stop("Unsupported architecture: ", architecture)
  )
}

#' @keywords internal
.vgg_logits_weights <- function(model) {
  layer <- keras3::get_layer(model, name = "predictions")
  w <- layer$get_weights()
  list(kernel = as.matrix(w[[1]]), bias = as.numeric(w[[2]]))
}

#' @noRd
.dense_logits <- function(fc_activations, kernel, bias) {
  X <- as.matrix(fc_activations)
  storage.mode(X) <- "double"
  kernel <- as.matrix(kernel)
  out <- sweep(X %*% kernel, 2L, as.numeric(bias), `+`)
  storage.mode(out) <- "double"
  out
}

#' @keywords internal
.resolve_vgg_taps <- function(taps) {
  keras_taps <- taps
  logits_stages <- names(taps)[unname(taps) == "classifier_logits"]
  if (length(logits_stages)) {
    keras_taps[logits_stages] <- "fc2"
  }
  list(
    user_taps = taps,
    keras_taps = keras_taps,
    logits_stages = logits_stages,
    unique_layers = unique(unname(keras_taps))
  )
}
