library(testthat)
# library(imfeatures)

context(".process_feature_map helper function")
# with_mocked_bindings defined in helper-mocks.R

test_that("average pooling returns correct values for 4D input", {
  # Create a dummy feature map: batch=1, H=2, W=2, C=2
  p <- array(0, dim = c(1, 2, 2, 2))
  # Channel 1 values: 1,2,3,4 -> mean = 2.5
  p[1, , , 1] <- matrix(c(1, 2, 3, 4), nrow = 2, byrow = TRUE)
  # Channel 2 values: 5,6,7,8 -> mean = 6.5
  p[1, , , 2] <- matrix(c(5, 6, 7, 8), nrow = 2, byrow = TRUE)

  out <- imfeatures:::.process_feature_map(p, "avg")
  expect_equal(out, c(2.5, 6.5))
})

test_that("maximum pooling returns correct values for 4D input", {
  p <- array(0, dim = c(1, 2, 2, 2))
  p[1, , , 1] <- matrix(c(1, 2, 3, 4), nrow = 2, byrow = TRUE) # max = 4
  p[1, , , 2] <- matrix(c(5, 6, 7, 8), nrow = 2, byrow = TRUE) # max = 8

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

test_that("resize downsampling uses area-pooled cell means (regression #48)", {
  # Skip if TensorFlow is not available
  skip_if_not_installed("tensorflow")
  skip_if_not(
    reticulate::py_module_available("tensorflow"),
    "TensorFlow Python module not available"
  )

  # A 6x6 map reduced to 2x2 has four non-overlapping 3x3 cells.
  # Area pooling must return each cell mean rather than point samples.
  p <- array(seq_len(36), dim = c(1, 6, 6, 1))

  # Resize to 2x2 and flatten
  out <- imfeatures:::.process_feature_map(p, "resize_2x2")
  expected <- c(
    mean(p[1, 1:3, 1:3, 1]),
    mean(p[1, 4:6, 1:3, 1]),
    mean(p[1, 1:3, 4:6, 1]),
    mean(p[1, 4:6, 4:6, 1])
  )

  expect_type(out, "double")
  expect_equal(out, expected, tolerance = 1e-6)

  # Averaging the area-pooled grid must reproduce global average pooling,
  # including when source and target sizes are not integer multiples.
  p_irregular <- array(seq_len(2 * 7 * 5), dim = c(1, 7, 5, 2))
  resized <- imfeatures:::.process_feature_map(p_irregular, "resize_3x2")
  resized <- array(resized, dim = c(1, 3, 2, 2))
  expect_equal(
    as.vector(apply(resized, MARGIN = c(1, 4), FUN = mean)),
    imfeatures:::.process_feature_map(p_irregular, "avg"),
    tolerance = 1e-5
  )

  # Test invalid resize format still works
  p2 <- array(1:8, dim = c(1, 2, 2, 2))
  expect_warning(
    out2 <- imfeatures:::.process_feature_map(p2, "resize_invalid"),
    "Invalid resize format"
  )
  expect_identical(out2, p2)
})
