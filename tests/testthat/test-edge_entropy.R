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

test_that("compute_edge_entropy matches between R and C++ on deterministic matrices", {
  compare_impls <- function(img, tol = 5e-4) {
    result_r <- compute_edge_entropy(img, use_cpp = FALSE)
    result_cpp <- compute_edge_entropy(img, use_cpp = TRUE)

    expect_identical(result_cpp$im, result_r$im)
    expect_equal(result_cpp$entropy, result_r$entropy, tolerance = tol)
    expect_equal(result_cpp$pentropy_20_80, result_r$pentropy_20_80, tolerance = tol)
    expect_equal(result_cpp$pentropy_80_160, result_r$pentropy_80_160, tolerance = tol)
    expect_equal(result_cpp$pentropy_160_240, result_r$pentropy_160_240, tolerance = tol)
    expect_equal(result_cpp$complex_before, result_r$complex_before, tolerance = tol)
  }

  set.seed(1)
  compare_impls(matrix(runif(50 * 60), nrow = 50))

  sparse_large <- matrix(0, nrow = 101, ncol = 100)
  sparse_large[49:53, 48:52] <- 1
  compare_impls(sparse_large, tol = 5e-4)
})
