library(testthat)
library(imfeatures)

context("edge_entropy filter_length validation")

img_matrix <- matrix(1:9, nrow = 3)

# C++ branch
expect_error(
  edge_entropy(img_matrix, filter_length = 2, use_cpp = TRUE),
  "'filter_length' must be a positive odd integer"
)

expect_error(
  edge_entropy(img_matrix, filter_length = -1, use_cpp = TRUE),
  "'filter_length' must be a positive odd integer"
)

# R branch
expect_error(
  edge_entropy(img_matrix, filter_length = 2, use_cpp = FALSE),
  "'filter_length' must be a positive odd integer"
)

expect_error(
  edge_entropy(img_matrix, filter_length = -1, use_cpp = FALSE),
  "'filter_length' must be a positive odd integer"
)
