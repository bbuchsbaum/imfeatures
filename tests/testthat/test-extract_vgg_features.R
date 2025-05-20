library(testthat)
library(imfeatures)

context("extract_vgg_features")

# mock helpers -----------------------------------------------------------
mock_im_features <- function(impath, layers, model = NULL, target_size = c(224,224), spatial_pooling = "avg") {
  lapply(seq_along(layers), function(i) c(i, i))
}

mock_get_layer <- function(model, index) {
  list(name = paste0("L", index))
}

make_dummy_images <- function(dir, n = 2) {
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  paths <- file.path(dir, paste0("img", seq_len(n), ".png"))
  for (p in paths) file.create(p)
  paths
}

# Tests -----------------------------------------------------------------

test_that("directory input expands to images and returns expected dims", {
  img_dir <- file.path(tempdir(), "imgs")
  make_dummy_images(img_dir, 2)
  with_mock(
    `imfeatures::im_features` = mock_im_features,
    `keras::get_layer` = mock_get_layer,
    {
      res <- extract_vgg_features(img_dir, tier = "low", model = list(dummy=TRUE))
      expect_s3_class(res, "vgg_feature_set")
      expect_equal(length(res$image_paths), 2)
      expect_equal(nrow(res$features), 2)
      expect_equal(ncol(res$features), length(res$layer_names) * 2)
    }
  )
})

test_that("error for nonexistent image paths", {
  expect_error(
    extract_vgg_features(c("no_such_file1.png", "no_such_file2.png"), model = list(dummy=TRUE)),
    "do not exist"
  )
})

test_that("returns correct class and dims for explicit image paths", {
  img_dir <- file.path(tempdir(), "imgs2")
  paths <- make_dummy_images(img_dir, 3)
  with_mock(
    `imfeatures::im_features` = mock_im_features,
    `keras::get_layer` = mock_get_layer,
    {
      res <- extract_vgg_features(paths[1:2], tier = "low", model = list(dummy=TRUE))
      expect_s3_class(res, "vgg_feature_set")
      expect_equal(nrow(res$features), 2)
      expect_equal(ncol(res$features), length(res$layer_names) * 2)
    }
  )
})
