library(testthat)

context("hierarchy transform: inductive residualization")

make_collinear_blocks <- function(n = 120, p1 = 12, p2 = 10, seed = 1) {
  set.seed(seed)
  b1 <- matrix(rnorm(n * p1), n, p1)
  colnames(b1) <- sprintf("a%02d", seq_len(p1))
  coef <- matrix(rnorm(p1 * p2, sd = 0.8), p1, p2)
  b2 <- b1 %*% coef + matrix(rnorm(n * p2, sd = 0.25), n, p2)
  colnames(b2) <- sprintf("b%02d", seq_len(p2))
  list(b1 = b1, b2 = b2)
}

standard_recipe <- function(rank = 6, name_prefix = "vgg",
                            post_rank = NULL, method = "svd", lambda = 0) {
  hierarchy_recipe(
    order = c("b1", "b2"),
    compress = block_pca(
      rank = rank,
      feature_scale = "sd",
      score_weighting = "whitened"
    ),
    decomposition = ordered_innovation(method = method, lambda = lambda),
    post_compress = if (is.null(post_rank)) {
      NULL
    } else {
      block_pca(
        rank = post_rank,
        feature_scale = "none",
        score_weighting = "whitened"
      )
    },
    name_prefix = name_prefix
  )
}

test_that("fit stores coefficient maps and predict reconstructs training residuals", {
  blocks <- make_collinear_blocks()
  fit <- fit_hierarchy_transform(standard_recipe(), blocks)
  expect_s3_class(fit, "hierarchy_transform")
  expect_equal(unname(fit$output_names), c("vgg_b1", "vgg_b2_innovation"))
  expect_equal(nrow(fit$mapping$b2$B_hat), ncol(fit$training_scores$b1))
  expect_equal(ncol(fit$mapping$b2$B_hat), ncol(fit$training_scores$b2))

  pred <- predict(fit, blocks)
  expect_equal(names(pred), c("vgg_b1", "vgg_b2_innovation"))
  expect_equal(pred$vgg_b1, fit$residuals$vgg_b1, tolerance = 1e-10)
  expect_equal(pred$vgg_b2_innovation, fit$residuals$vgg_b2_innovation, tolerance = 1e-10)
  expect_identical(apply_hierarchy_transform(fit, blocks), pred)
})

test_that("training innovations are orthogonal for unregularized SVD", {
  blocks <- make_collinear_blocks()
  fit <- fit_hierarchy_transform(standard_recipe(), blocks)
  e1 <- fit$training_innovations$b1
  e2 <- fit$training_innovations$b2
  expect_lt(max(abs(crossprod(e1, e2))), 1e-8)
})

test_that("held-out orthogonality is not required", {
  train <- make_collinear_blocks(n = 80, seed = 2)
  test <- make_collinear_blocks(n = 80, seed = 3)
  fit <- fit_hierarchy_transform(standard_recipe(), train)
  pred <- predict(fit, test)
  orth <- crossprod(pred$vgg_b1, pred$vgg_b2_innovation)
  expect_true(is.finite(max(abs(orth))))
})

test_that("transform is batch-invariant and row-permutation equivariant", {
  train <- make_collinear_blocks(n = 90, seed = 4)
  test <- make_collinear_blocks(n = 40, seed = 5)
  fit <- fit_hierarchy_transform(standard_recipe(), train)
  pred_all <- predict(fit, test)
  idx <- c(3L, 8L, 21L, 22L, 39L)
  pred_sub <- predict(fit, subset_feature_blocks(test, idx))
  expect_equal(pred_all$vgg_b1[idx, , drop = FALSE], pred_sub$vgg_b1, tolerance = 1e-10)
  expect_equal(
    pred_all$vgg_b2_innovation[idx, , drop = FALSE],
    pred_sub$vgg_b2_innovation,
    tolerance = 1e-10
  )

  perm <- sample.int(40)
  pred_perm <- predict(fit, subset_feature_blocks(test, perm))
  expect_equal(pred_perm$vgg_b1, pred_all$vgg_b1[perm, , drop = FALSE], tolerance = 1e-10)
  expect_equal(
    pred_perm$vgg_b2_innovation,
    pred_all$vgg_b2_innovation[perm, , drop = FALSE],
    tolerance = 1e-10
  )
})

test_that("extra test rows cannot change already-transformed rows", {
  train <- make_collinear_blocks(n = 60, seed = 6)
  test <- make_collinear_blocks(n = 30, seed = 7)
  extra <- make_collinear_blocks(n = 15, seed = 8)
  fit <- fit_hierarchy_transform(standard_recipe(), train)
  pred_test <- predict(fit, test)
  combined <- list(
    b1 = rbind(test$b1, extra$b1),
    b2 = rbind(test$b2, extra$b2)
  )
  pred_combined <- predict(fit, combined)
  expect_equal(pred_combined$vgg_b1[seq_len(30), , drop = FALSE], pred_test$vgg_b1, tolerance = 1e-10)
  expect_equal(
    pred_combined$vgg_b2_innovation[seq_len(30), , drop = FALSE],
    pred_test$vgg_b2_innovation,
    tolerance = 1e-10
  )
})

test_that("output schema is locked on new data", {
  train <- make_collinear_blocks()
  test <- make_collinear_blocks(n = 25, seed = 9)
  fit <- fit_hierarchy_transform(standard_recipe(rank = 5, post_rank = 4), train)
  pred <- predict(fit, test)
  expect_equal(names(pred), names(fit$residuals))
  expect_equal(vapply(pred, ncol, integer(1)), vapply(fit$residuals, ncol, integer(1)))
  expect_equal(colnames(pred$vgg_b2_innovation), colnames(fit$residuals$vgg_b2_innovation))
})

test_that("column reordering by name is recovered", {
  train <- make_collinear_blocks()
  test <- make_collinear_blocks(n = 20, seed = 10)
  fit <- fit_hierarchy_transform(standard_recipe(), train)
  shuffled <- test
  shuffled$b1 <- shuffled$b1[, rev(colnames(shuffled$b1)), drop = FALSE]
  shuffled$b2 <- shuffled$b2[, rev(colnames(shuffled$b2)), drop = FALSE]
  expect_equal(predict(fit, shuffled), predict(fit, test), tolerance = 1e-10)
})

test_that("serialization round-trip preserves predict", {
  train <- make_collinear_blocks()
  test <- make_collinear_blocks(n = 18, seed = 11)
  fit <- fit_hierarchy_transform(standard_recipe(), train)
  tf <- tempfile(fileext = ".rds")
  on.exit(unlink(tf), add = TRUE)
  saveRDS(fit, tf)
  restored <- readRDS(tf)
  expect_equal(predict(restored, test), predict(fit, test), tolerance = 1e-10)
})

test_that("known linear mapping is recovered by SVD residualization", {
  set.seed(12)
  n <- 80
  p_h <- 5
  p_z <- 4
  H <- matrix(rnorm(n * p_h), n, p_h)
  B <- matrix(rnorm(p_h * p_z), p_h, p_z)
  E <- matrix(rnorm(n * p_z), n, p_z)
  E <- E - H %*% solve(crossprod(H), crossprod(H, E))
  Z <- H %*% B + E
  map <- .fit_stage_mapping(H, Z, ordered_innovation(method = "svd"))
  expect_equal(map$B_hat, B, tolerance = 1e-8)
  expect_equal(.apply_stage_mapping(H, Z, map$B_hat), E, tolerance = 1e-8)
})

test_that("exact linear dependence yields near-zero held-out innovation", {
  set.seed(13)
  n <- 100
  p <- 8
  b1 <- matrix(rnorm(n * p), n, p)
  colnames(b1) <- paste0("x", seq_len(p))
  B <- matrix(rnorm(p * 6), p, 6)
  b2 <- b1 %*% B
  colnames(b2) <- paste0("y", seq_len(6))
  recipe <- hierarchy_recipe(
    order = c("b1", "b2"),
    compress = block_pca(rank = 8, feature_scale = "none", score_weighting = "singular"),
    decomposition = ordered_innovation(method = "svd")
  )
  fit <- fit_hierarchy_transform(recipe, list(b1 = b1, b2 = b2))
  expect_lt(max(abs(fit$training_innovations$b2)), 1e-8)

  b1_new <- matrix(rnorm(40 * p), 40, p)
  colnames(b1_new) <- colnames(b1)
  b2_new <- b1_new %*% B
  colnames(b2_new) <- colnames(b2)
  pred <- predict(fit, list(b1 = b1_new, b2 = b2_new))
  expect_lt(max(abs(pred$b2_innovation)), 1e-7)
})

test_that("ridge with large lambda leaves later stages almost unresidualized", {
  blocks <- make_collinear_blocks()
  fit0 <- fit_hierarchy_transform(standard_recipe(method = "ridge", lambda = 1e8), blocks)
  z2 <- fit0$training_scores$b2
  e2 <- fit0$training_innovations$b2
  expect_lt(max(abs(e2 - z2)), 1e-4)
})

test_that("blocked_cv is rejected at fit time", {
  blocks <- make_collinear_blocks()
  recipe <- hierarchy_recipe(
    order = c("b1", "b2"),
    compress = block_pca(rank = 4),
    decomposition = ordered_innovation(method = "ridge", lambda = "blocked_cv")
  )
  expect_error(
    fit_hierarchy_transform(recipe, blocks),
    "blocked_cv"
  )
})

test_that("feature_hierarchy subset keeps sample identifiers", {
  blocks <- make_collinear_blocks(n = 10)
  hier <- as_feature_hierarchy(blocks, sample_id = letters[1:10])
  sub <- subset_feature_blocks(hier, c(2L, 5L, 9L))
  expect_s3_class(sub, "feature_hierarchy")
  expect_equal(sub$sample_id, c("b", "e", "i"))
  expect_equal(nrow(sub$blocks$b1), 3)
})

test_that("non-finite and row-mismatch inputs fail", {
  expect_error(
    as_feature_hierarchy(list(b1 = matrix(c(1, NA, 3, 4), 2), b2 = matrix(rnorm(4), 2))),
    "non-finite"
  )
  expect_error(
    as_feature_hierarchy(list(b1 = matrix(rnorm(6), 3), b2 = matrix(rnorm(4), 2))),
    "same number of rows"
  )
})

test_that("eigencore compression matches stats on a small problem", {
  skip_if_not_installed("eigencore")
  blocks <- make_collinear_blocks(n = 60, p1 = 8, p2 = 8, seed = 14)
  rec_stats <- hierarchy_recipe(
    order = c("b1", "b2"),
    compress = block_pca(rank = 4, method = "stats", score_weighting = "singular"),
    decomposition = ordered_innovation(method = "svd")
  )
  rec_eigen <- hierarchy_recipe(
    order = c("b1", "b2"),
    compress = block_pca(rank = 4, method = "eigencore", score_weighting = "singular"),
    decomposition = ordered_innovation(method = "svd")
  )
  fit_s <- fit_hierarchy_transform(rec_stats, blocks)
  fit_e <- fit_hierarchy_transform(rec_eigen, blocks)
  align <- function(ref, x) {
    for (j in seq_len(ncol(ref))) {
      if (sum(ref[, j] * x[, j]) < 0) x[, j] <- -x[, j]
    }
    x
  }
  aligned <- align(unname(fit_s$training_scores$b1), unname(fit_e$training_scores$b1))
  expect_lt(max(abs(aligned - unname(fit_s$training_scores$b1))), 1e-8)
})
