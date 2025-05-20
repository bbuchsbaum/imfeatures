library(testthat)
library(imfeatures)

context("image_mse")

# Helper to create a simple grayscale cimg
make_gray <- function(vals) {
  imager::as.cimg(matrix(vals, nrow = 2, byrow = TRUE))
}

# Helper to create a simple 2x2 RGB cimg
make_color <- function(pixels) {
  arr <- array(pixels, dim = c(2, 2, 1, 3))
  imager::as.cimg(arr)
}

# ---------------------------------------------------------------------
# Grayscale image should be internally converted and return 3 values
# ---------------------------------------------------------------------

test_that("grayscale images return numeric vector of length 3", {
  img <- make_gray(c(0, 0, 1, 1))
  res <- image_mse(img, sf = 0, bins = 2)
  expect_type(res, "double")
  expect_length(res, 3)
  expect_true(!anyNA(res))
})

# ---------------------------------------------------------------------
# Color image handling
# ---------------------------------------------------------------------

test_that("color images return numeric vector of length 3", {
  # Red, green, blue, yellow pixels
  pix <- c(1,0,0, 0,1,0, 0,0,1, 1,1,0)
  img <- make_color(pix)
  res <- image_mse(img, sf = 0, bins = 2)
  expect_type(res, "double")
  expect_length(res, 3)
  expect_true(!anyNA(res))
})

# ---------------------------------------------------------------------
# NA handling
# ---------------------------------------------------------------------

test_that("NA values lead to NA entropy without warnings", {
  img <- make_gray(c(1, NA, 0, 1))
  expect_warning(res <- image_mse(img, sf = 0, bins = 2), regexp = NA)
  expect_true(anyNA(res))
})

# ---------------------------------------------------------------------
# Incorrect channel counts
# ---------------------------------------------------------------------

test_that("images with incorrect channel counts throw an error", {
  bad_img <- imager::as.cimg(array(runif(2*2*1*4), dim = c(2,2,1,4)))
  expect_error(image_mse(bad_img))
})

