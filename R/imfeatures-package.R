#' imfeatures: Extract Visual Features from 2D Images
#'
#' Tools for extracting visual features from 2D images using deep learning
#' models and traditional computer vision techniques. Includes support for
#' VGG16, ResNet, and other pre-trained models through Python integration,
#' as well as methods for computing edge entropy, multiscale entropy, and
#' feature similarity metrics.
#'
#' @keywords internal
#' @importFrom stats predict
#' @importFrom utils data head
#' @importFrom dplyr %>%
#' @useDynLib imfeatures, .registration = TRUE
"_PACKAGE"

utils::globalVariables(c(
  "places_cat365",
  "score"
))
