library(testthat)

context("compute_edge_entropy")

test_that("compute_edge_entropy filter_length validation works", {
  img_matrix <- matrix(1:9, nrow = 3)
  
  # C++ branch
  expect_error(
    compute_edge_entropy(img_matrix, filter_length = 2L, use_cpp = TRUE),
    "'filter_length' must be a positive odd integer"
  )
  
  expect_error(
    compute_edge_entropy(img_matrix, filter_length = -1L, use_cpp = TRUE),
    "'filter_length' must be a positive odd integer"
  )
  
  # R branch
  expect_error(
    compute_edge_entropy(img_matrix, filter_length = 2L, use_cpp = FALSE),
    "'filter_length' must be a positive odd integer"
  )
  
  expect_error(
    compute_edge_entropy(img_matrix, filter_length = -1L, use_cpp = FALSE),
    "'filter_length' must be a positive odd integer"
  )
})
