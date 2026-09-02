library(testthat)

context("hierarchy diagnostics")

test_that("linear CKA is 1 for identical matrices and symmetric", {
  set.seed(1)
  X <- matrix(rnorm(40 * 5), 40, 5)
  Y <- matrix(rnorm(40 * 3), 40, 3)
  expect_equal(linear_cka(X, X), 1, tolerance = 1e-10)
  expect_equal(linear_cka(X, Y), linear_cka(Y, X), tolerance = 1e-12)
  expect_gte(linear_cka(X, Y), 0)
  expect_lte(linear_cka(X, Y), 1)
})

test_that("effective rank is 1 for a rank-1 matrix", {
  x <- seq_len(20)
  X <- cbind(x, 2 * x, 3 * x)
  expect_equal(effective_rank(X), 1, tolerance = 1e-8)
})

test_that("training residual CKA is near zero after ordered innovation", {
  set.seed(2)
  blocks <- list(
    b1 = matrix(rnorm(80 * 10), 80, 10),
    b2 = matrix(rnorm(80 * 10), 80, 10)
  )
  colnames(blocks$b1) <- paste0("a", 1:10)
  colnames(blocks$b2) <- paste0("b", 1:10)
  fit <- fit_hierarchy_transform(
    hierarchy_recipe(
      order = c("b1", "b2"),
      compress = block_pca(rank = 6),
      decomposition = ordered_innovation(method = "svd")
    ),
    blocks
  )
  diag <- hierarchy_diagnostics(fit, blocks)
  expect_s3_class(diag, "hierarchy_diagnostics")
  expect_lt(abs(diag$pairwise_cka["b1", "b2"]), 1e-8)
  expect_true(all(diag$innovation$residual_energy >= 0, na.rm = TRUE))
  expect_true(all(diag$innovation$residual_energy <= 1 + 1e-8, na.rm = TRUE))
  expect_true(all(diag$compression$cka_retained > 0.8, na.rm = TRUE))
})

test_that("held-out residual CKA can be positive", {
  set.seed(3)
  train <- list(
    b1 = matrix(rnorm(70 * 8), 70, 8),
    b2 = matrix(rnorm(70 * 8), 70, 8)
  )
  test <- list(
    b1 = matrix(rnorm(40 * 8), 40, 8),
    b2 = matrix(rnorm(40 * 8), 40, 8)
  )
  fit <- fit_hierarchy_transform(
    hierarchy_recipe(
      order = c("b1", "b2"),
      compress = block_pca(rank = 4),
      decomposition = ordered_innovation(method = "svd")
    ),
    train
  )
  diag <- hierarchy_diagnostics(fit, train, heldout = test)
  expect_true(!is.null(diag$heldout$pairwise_cka))
  expect_true(is.finite(diag$heldout$pairwise_cka["b1", "b2"]))
})

test_that("check_transform_invariance passes for an inductive fit", {
  set.seed(4)
  train <- list(
    b1 = matrix(rnorm(50 * 6), 50, 6),
    b2 = matrix(rnorm(50 * 6), 50, 6)
  )
  test <- list(
    b1 = matrix(rnorm(24 * 6), 24, 6),
    b2 = matrix(rnorm(24 * 6), 24, 6)
  )
  colnames(train$b1) <- colnames(test$b1) <- paste0("x", 1:6)
  colnames(train$b2) <- colnames(test$b2) <- paste0("y", 1:6)
  fit <- fit_hierarchy_transform(
    hierarchy_recipe(
      order = c("b1", "b2"),
      compress = block_pca(rank = 3),
      decomposition = ordered_innovation(method = "svd")
    ),
    train
  )
  chk <- check_transform_invariance(fit, test, tolerance = 1e-8)
  expect_true(chk$batch_invariant)
  expect_true(chk$permutation_equivariant)
})
