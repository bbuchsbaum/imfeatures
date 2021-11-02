
#' @export
#' @import reticulate
vgg16_places <- function() {
  m <- reticulate::import("keras_models.models.pretrained.vgg16_places365")
  model <- m$VGG16_Places365()
}

# place_features <- function(impath, layers, model=NULL, target_size=c(224,224)) {
#   if (is.null(model)) {
#     m <- reticulate::import("keras_models.models.pretrained.vgg16_places365")
#     model <- m$VGG16_Places365()
#   }
#
#   img <- image_load(impath, target_size = target_size)
#
#   x <- image_to_array(img)
#   x <- array_reshape(x, c(1, dim(x)))
#   #x <- array_reshape(x, c(1, unlist(x$shape)))
#   x <- imagenet_preprocess_input(x)
#
#   features <- lapply(layers, function(index) {
#     intermediate_layer_model <- keras_model(inputs = model$input,
#                                             outputs = get_layer(model, index=index)$output)
#
#     predict(intermediate_layer_model, x)
#   })
#
#   #preds <- model %>% predict(x,6)
#   #preds <- imagenet_decode_predictions(preds,10)
#   features
# }
