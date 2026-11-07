library(testthat)
library(imfeatures)

context("im_feature_sim")

test_that("compute_feature_similarity validates inputs correctly", {
  # Create temporary test images for validation tests
  temp_dir <- tempdir()

  # Test single image error (needs an actual file to pass assert_image)
  single_img <- file.path(temp_dir, "single_test.png")
  png(single_img, width = 100, height = 100)
  par(mar = c(0, 0, 0, 0))
  image(matrix(runif(100), 10, 10), axes = FALSE)
  dev.off()

  expect_error(
    compute_feature_similarity(single_img, layers = "block5_pool"),
    "need at least two images"
  )
  unlink(single_img)

  # Test subsamp_prop validation (needs actual files to pass assert_image)
  temp_imgs <- file.path(temp_dir, paste0("val_test_", 1:2, ".png"))
  for (img in temp_imgs) {
    png(img, width = 100, height = 100)
    par(mar = c(0, 0, 0, 0))
    image(matrix(runif(100), 10, 10), axes = FALSE)
    dev.off()
  }

  expect_error(
    compute_feature_similarity(temp_imgs, layers = "fc1", subsamp_prop = 1.5),
    "Assertion on 'subsamp_prop' failed"
  )
  unlink(temp_imgs)
})

test_that("im_feature_sim is an alias for compute_feature_similarity", {
  # Test that the alias exists and is identical
  expect_identical(im_feature_sim, compute_feature_similarity)
})

# Integration test that requires Keras - skip if not available
test_that("compute_feature_similarity works end-to-end with real model", {
  skip_if_not_installed("keras3")
  skip_if_not(
    reticulate::py_module_available("keras"),
    "Keras Python module not available"
  )

  # Try to load VGG16 - skip if it fails (e.g., TensorFlow config issues)
  skip_on_cran() # Don't run this heavy test on CRAN

  model_load_works <- tryCatch(
    {
      model <- keras3::application_vgg16(weights = "imagenet", include_top = TRUE)
      TRUE
    },
    error = function(e) {
      FALSE
    }
  )

  skip_if_not(model_load_works, "VGG16 model cannot be loaded")

  # Create temporary test images
  temp_dir <- tempdir()
  img_paths <- file.path(temp_dir, paste0("test_img_", 1:3, ".png"))

  for (i in seq_along(img_paths)) {
    png(img_paths[i], width = 224, height = 224)
    par(mar = c(0, 0, 0, 0))
    image(matrix(runif(224 * 224), 224, 224), axes = FALSE, col = rainbow(10))
    dev.off()
  }

  # Run the actual similarity computation
  res <- compute_feature_similarity(
    img_paths,
    layers = "block5_pool",
    lowmem = TRUE
  )

  # Check structure
  expect_type(res, "list")
  expect_length(res, 1)
  expect_true("layer_block5_pool" %in% names(res))

  # Check matrix properties
  sim_matrix <- res[[1]]
  expect_true(is.matrix(sim_matrix))
  expect_equal(dim(sim_matrix), c(3, 3))
  expect_true(isSymmetric(sim_matrix))

  # Check that diagonal elements are close to 1 (self-similarity)
  expect_true(all(diag(sim_matrix) > 0.99))

  # Check that off-diagonal elements are valid similarities
  expect_true(all(sim_matrix >= -1 & sim_matrix <= 1))

  # Clean up
  unlink(img_paths)
})

test_that("low-memory similarity uses feature values and fills the diagonal", {
  temp_imgs <- file.path(tempdir(), paste0("sim_mock_", 1:3, ".png"))
  for (img in temp_imgs) {
    png(img, width = 10, height = 10)
    par(mar = c(0, 0, 0, 0))
    image(matrix(runif(100), 10, 10), axes = FALSE)
    dev.off()
  }
  on.exit(unlink(temp_imgs))
  feats <- list(c(1, 0, 0, 1), c(1, 1, 0, 0), c(-1, 0, 2, 1))
  names(feats) <- temp_imgs
  # Same tibble layout as extract_features(): image, layer, feature
  fake_extract <- function(impath, layers, model = NULL, spatial_pooling = "none", ...) {
    tibble::tibble(image = impath, layer = layers, feature = list(feats[[impath]]))
  }
  testthat::local_mocked_bindings(extract_features = fake_extract)

  res <- compute_feature_similarity(temp_imgs, layers = "fc1", model = "mock", lowmem = TRUE)
  sim <- res$layer_fc1
  cosine <- function(a, b) sum(a * b) / sqrt(sum(a^2) * sum(b^2))
  expected <- outer(1:3, 1:3, Vectorize(function(i, j) cosine(feats[[i]], feats[[j]])))
  expect_equal(unname(sim), expected, tolerance = 1e-12)
  expect_equal(unname(diag(sim)), rep(1, 3))
  expect_true(isSymmetric(sim))
})
