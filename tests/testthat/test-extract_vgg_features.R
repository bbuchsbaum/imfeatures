library(testthat)

context("extract_vgg_features")

mock_im_features <- function(impath, layers, model = NULL,
                             target_size = c(224, 224),
                             spatial_pooling = "avg") {
  tibble::tibble(
    image = impath,
    layer = layers,
    feature = lapply(seq_along(layers), function(i) c(i, i) + 0.0)
  )
}

make_dummy_images <- function(dir, n = 2) {
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  paths <- file.path(dir, paste0("img", seq_len(n), ".png"))
  for (p in paths) file.create(p)
  paths
}

test_that("directory input expands to images and returns expected dims", {
  local_mocked_bindings(im_features = mock_im_features, .package = "imfeatures")
  img_dir <- file.path(tempdir(), "imgs")
  make_dummy_images(img_dir, 2)
  res <- extract_vgg_features(img_dir, tier = "low", model = list(dummy = TRUE))
  expect_s3_class(res, "vgg_feature_set")
  expect_equal(length(res$image_paths), 2)
  expect_equal(nrow(res$features), 2)
  expect_equal(ncol(res$features), length(res$layer_names) * 2)
})

test_that("error for nonexistent image paths", {
  expect_error(
    extract_vgg_features(c("no_such_file1.png", "no_such_file2.png"),
                        model = list(dummy = TRUE)),
    "impaths file\\(s\\) not found: no_such_file1.png, no_such_file2.png"
  )
})

test_that("returns correct class and dims for explicit image paths", {
  local_mocked_bindings(im_features = mock_im_features, .package = "imfeatures")
  img_dir <- file.path(tempdir(), "imgs2")
  paths <- make_dummy_images(img_dir, 3)
  res <- extract_vgg_features(paths[1:2], tier = "low", model = list(dummy = TRUE))
  expect_s3_class(res, "vgg_feature_set")
  expect_equal(nrow(res$features), 2)
  expect_equal(ncol(res$features), length(res$layer_names) * 2)
})

test_that("features matrix is numeric, not character (regression #45)", {
  local_mocked_bindings(im_features = mock_im_features, .package = "imfeatures")
  img_dir <- file.path(tempdir(), "imgs_numeric")
  paths <- make_dummy_images(img_dir, 2)
  res <- extract_vgg_features(paths, tier = "low", model = list(dummy = TRUE))
  expect_true(is.numeric(res$features))
  expect_equal(typeof(res$features), "double")
  expect_false(any(is.character(res$features)))
  expect_equal(nrow(res$features), length(paths))
})
