library(testthat)
#library(imfeatures)

context(".process_feature_map helper function")
# with_mocked_bindings defined in helper-mocks.R

test_that("average pooling returns correct values for 4D input", {
  # Create a dummy feature map: batch=1, H=2, W=2, C=2
  p <- array(0, dim = c(1, 2, 2, 2))
  # Channel 1 values: 1,2,3,4 -> mean = 2.5
  p[1, , , 1] <- matrix(c(1,2,3,4), nrow = 2, byrow = TRUE)
  # Channel 2 values: 5,6,7,8 -> mean = 6.5
  p[1, , , 2] <- matrix(c(5,6,7,8), nrow = 2, byrow = TRUE)

  out <- imfeatures:::.process_feature_map(p, "avg")
  expect_equal(out, c(2.5, 6.5))
})

test_that("maximum pooling returns correct values for 4D input", {
  p <- array(0, dim = c(1, 2, 2, 2))
  p[1, , , 1] <- matrix(c(1,2,3,4), nrow = 2, byrow = TRUE) # max = 4
  p[1, , , 2] <- matrix(c(5,6,7,8), nrow = 2, byrow = TRUE) # max = 8

  out <- imfeatures:::.process_feature_map(p, "max")
  expect_equal(out, c(4, 8))
})

test_that("no pooling returns the original 4D array", {
  p <- array(1:24, dim = c(1, 2, 3, 4))
  out <- imfeatures:::.process_feature_map(p, "none")
  expect_identical(out, p)
})

# Test that non-4D inputs are returned unchanged
test_that("non-4D input is returned unchanged", {
  p2d <- matrix(1:6, nrow = 2)
  out2d <- imfeatures:::.process_feature_map(p2d, "avg")
  expect_identical(out2d, p2d)
})

# Test invalid resize option returns original with warning
test_that("invalid resize option returns original with warning", {
  p <- array(1:8, dim = c(1, 2, 2, 1))
  expect_warning(
    out <- imfeatures:::.process_feature_map(p, "resize_abc"),
    "Invalid resize format"
  )
  expect_identical(out, p)
})

test_that("resize option calls tensorflow and returns flattened output", {
  p <- array(1:16, dim = c(1, 4, 4, 1))
  fake_tf <- list(
    constant = function(x, dtype) x,
    float32 = "float32",
    image = list(
      resize = function(images, size, method) {
        array(seq_len(size[[1]] * size[[2]]),
              dim = c(1, size[[1]], size[[2]], 1))
      },
      ResizeMethod = list(BILINEAR = "bilinear")
    )
  )
  out <- with_mocked_bindings(
    requireNamespace = function(pkg, quietly = TRUE) TRUE,
    `reticulate::import` = function(name, delay_load = TRUE) fake_tf,
    imfeatures:::.process_feature_map(p, "resize_2x2")
  )
  expect_equal(out, as.vector(array(1:4, dim = c(1, 2, 2, 1))))
})
