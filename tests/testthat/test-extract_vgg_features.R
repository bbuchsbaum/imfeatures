library(testthat)

context("extract_vgg_features")

# extract_vgg_features() runs one multi-output model over batches of images
# rather than one model per layer per image, so the tests substitute the two
# internal seams (model construction and the batched forward pass) instead of
# mocking im_features().

mock_multi_model <- function(model, layers) {
  structure(list(layers = layers), class = "mock_multi_model")
}

# One array per layer, first dimension indexing the image in the batch.
# Two features per layer keeps the expected width at length(layer_names) * 2.
mock_forward_batch <- function(multi, paths, target_size) {
  lapply(seq_along(multi$layers), function(i) {
    matrix(rep(c(i, i) + 0.0, each = length(paths)), nrow = length(paths), ncol = 2)
  })
}

local_vgg_mocks <- function(env = parent.frame()) {
  local_mocked_bindings(
    .vgg_multi_output_model = mock_multi_model,
    .vgg_forward_batch      = mock_forward_batch,
    .package = "imfeatures",
    .env = env
  )
}

make_dummy_images <- function(dir, n = 2) {
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  paths <- file.path(dir, paste0("img", seq_len(n), ".png"))
  for (p in paths) file.create(p)
  paths
}

test_that("directory input expands to images and returns expected dims", {
  local_vgg_mocks()
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
  local_vgg_mocks()
  img_dir <- file.path(tempdir(), "imgs2")
  paths <- make_dummy_images(img_dir, 3)
  res <- extract_vgg_features(paths[1:2], tier = "low", model = list(dummy = TRUE))
  expect_s3_class(res, "vgg_feature_set")
  expect_equal(nrow(res$features), 2)
  expect_equal(ncol(res$features), length(res$layer_names) * 2)
})

test_that("features matrix is numeric, not character (regression #45)", {
  local_vgg_mocks()
  img_dir <- file.path(tempdir(), "imgs_numeric")
  paths <- make_dummy_images(img_dir, 2)
  res <- extract_vgg_features(paths, tier = "low", model = list(dummy = TRUE))
  expect_true(is.numeric(res$features))
  expect_equal(typeof(res$features), "double")
  expect_false(any(is.character(res$features)))
  expect_equal(nrow(res$features), length(paths))
})

test_that("batching does not change the result (regression #46)", {
  local_vgg_mocks()
  img_dir <- file.path(tempdir(), "imgs_batch")
  paths <- make_dummy_images(img_dir, 5)
  one <- extract_vgg_features(paths, tier = "low", model = list(dummy = TRUE), batch_size = 1)
  big <- extract_vgg_features(paths, tier = "low", model = list(dummy = TRUE), batch_size = 32)
  expect_identical(one$features, big$features)
  expect_equal(nrow(one$features), 5)
})

test_that("batch_size is validated", {
  img_dir <- file.path(tempdir(), "imgs_bs")
  paths <- make_dummy_images(img_dir, 2)
  expect_error(
    extract_vgg_features(paths, tier = "low", model = list(dummy = TRUE), batch_size = 0)
  )
})
