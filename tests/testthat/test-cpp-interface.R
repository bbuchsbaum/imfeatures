context("edge_entropy_cpp input validation")

# Using small dummy matrix
img <- matrix(1:4, nrow = 2)

test_that("edge_entropy_cpp validates inputs", {
  edge_entropy_cpp <- imfeatures:::edge_entropy_cpp

  # invalid image shape
  expect_error(
    edge_entropy_cpp(matrix(numeric(0), nrow = 0, ncol = 0)),
    "Input image has zero dimensions"
  )

  # invalid maxdiag
  expect_error(
    edge_entropy_cpp(img, maxdiag = 0),
    "maxdiag must be positive"
  )

  # invalid gabor_bins
  expect_error(
    edge_entropy_cpp(img, gabor_bins = 0),
    "gabor_bins must be positive"
  )

  # invalid filter_length (even)
  expect_error(
    edge_entropy_cpp(img, filter_length = 2),
    "filter_length must be positive and odd"
  )

  # invalid circ_bins
  expect_error(
    edge_entropy_cpp(img, circ_bins = 0),
    "circ_bins must be positive"
  )

  # invalid ranges type
  expect_error(
    edge_entropy_cpp(img, ranges = 1),
    "Parameter 'ranges' must be a list"
  )
})
