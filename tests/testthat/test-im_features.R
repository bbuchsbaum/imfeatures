library(testthat)
library(imfeatures)

context("im_features")

# helper to mock dependencies

with_mocked_bindings <- function(..., .env = environment()) {
  withr::local_bindings(..., .env = .env)
}


test_that("im_features errors on missing file paths", {
  expect_error(
    with_mocked_bindings(
      image_load = function(path, target_size) {
        stop("File not found")
      },
      image_to_array = function(img) NULL,
      array_reshape = function(x, dim) NULL,
      imagenet_preprocess_input = function(x) x,
      keras_model = function(inputs, outputs) NULL,
      get_layer = function(model, index) list(output = NULL),
      predict = function(model, x) NULL,
      im_features("missing.jpg", layers = 1, model = list())
    ),
    "File not found"
  )
})


test_that("im_features passes spatial_pooling option", {
  called <- NULL
  result <- with_mocked_bindings(
    image_load = function(path, target_size) "img",
    image_to_array = function(img) array(1, dim = c(1, 1, 1)),
    array_reshape = function(x, dim) x,
    imagenet_preprocess_input = function(x) x,
    keras_model = function(inputs, outputs) NULL,
    get_layer = function(model, index) list(output = NULL),
    predict = function(model, x) array(1:4, dim = c(1, 2, 2, 1)),
    .process_feature_map = function(p, spatial_pooling) {
      called <<- spatial_pooling
      paste0("processed_", spatial_pooling)
    },
    im_features("dummy.jpg", layers = 1, model = list(), spatial_pooling = "avg")
  )

  expect_equal(called, "avg")
  expect_equal(result[[1]], "processed_avg")
})


