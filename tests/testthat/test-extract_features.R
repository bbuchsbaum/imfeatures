library(testthat)
#library(imfeatures)

context("im_features")

# Create a simple test image file
create_test_img <- function(filename, dir = tempdir()) {
  path <- file.path(dir, filename)
  img <- matrix(runif(64*64), 64, 64)
  # Write using R's internal bitmap functions
  png::writePNG(img, path)
  return(path)
}

test_that("im_features errors on missing file paths", {
  # Test with a non-existent file path
  expect_error(
    im_features("missing.jpg", layers = 1),
    regexp = "No such file or directory|Image path does not exist"  # Accept either error message
  )
})

test_that("spatial_pooling parameter in im_features function is validated", {
  # This test simply verifies that the spatial_pooling parameter validation works
  expect_error(
    im_features("dummy.jpg", layers = 1, spatial_pooling = "invalid_value"),
    "'spatial_pooling' must be 'none', 'avg', 'max', or 'resize_HxW'"
  )
})


