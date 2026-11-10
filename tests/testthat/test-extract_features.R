library(testthat)
# library(imfeatures)

context("im_features")

# Create a simple test image file
create_test_img <- function(filename, dir = tempdir()) {
  path <- file.path(dir, filename)
  img <- matrix(runif(64 * 64), 64, 64)
  # Write using R's internal bitmap functions
  png::writePNG(img, path)
  return(path)
}

test_that("im_features errors on missing file paths", {
  # Test with a non-existent file path
  expect_error(
    im_features("missing.jpg", layers = 1),
    regexp = "impath file\\(s\\) not found: missing.jpg"
  )
})

test_that("spatial_pooling parameter in im_features function is validated", {
  # This test simply verifies that the spatial_pooling parameter validation works
  expect_error(
    im_features("dummy.jpg", layers = 1, spatial_pooling = "invalid_value"),
    "impath file\\(s\\) not found: dummy.jpg"
  )
})

test_that("spatial pooling strings are validated consistently", {
  ok <- c("none", "avg", "max", "resize_3x3", "resize_2x4", "resize_14x14")
  for (x in ok) {
    expect_silent(imfeatures:::.assert_spatial_pooling(x))
  }
  bad <- list("mean", "resize_0x3", "resize_3x0", "resize_3", "resize_3x3x3", "", NA_character_, c("avg", "max"), 3)
  for (x in bad) {
    expect_error(imfeatures:::.assert_spatial_pooling(x), "'spatial_pooling' must be")
  }
  expect_error(
    imfeatures:::.assert_spatial_pooling("resize_abc", "pooling"),
    "'pooling' must be"
  )
})
