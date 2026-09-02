# -------------------------------------------------------------------------
#  UNIT TEST  — residualize_tiers()
# -------------------------------------------------------------------------
library(testthat)

context("residualize_tiers: PCA and hierarchical residualization")

set.seed(42)
tol <- 1e-8

n <- 500
p1 <- 30
p2 <- 40

Xlow <- matrix(rnorm(n * p1), n, p1)
colnames(Xlow) <- paste0("L", 1:p1)

set.seed(123)
coef_matrix <- matrix(rnorm(p1 * p2, sd = .7), p1, p2)
set.seed(42)

signal <- Xlow %*% coef_matrix
noise <- matrix(rnorm(n * p2, sd = .3), n, p2)
Xhigh <- signal + noise
colnames(Xhigh) <- paste0("H", 1:p2)

feature_list <- list(low = Xlow, high = Xhigh)
obj <- residualize_tiers(feature_list, numpcs = 10, pca_method = "stats")

test_that("dimensions are correct", {
  expect_s3_class(obj, "residualized_tiers")
  expect_s3_class(obj, "hierarchy_transform")
  expect_named(obj$residuals, c("low", "high"))
  expect_equal(ncol(obj$residuals$low), 10)
  expect_equal(ncol(obj$residuals$high), 10)
  expect_equal(nrow(obj$residuals$high), n)
})

test_that("low and residualized high are orthogonal (training data)", {
  expect_true(all(abs(colMeans(obj$residuals$low)) < tol))
  expect_true(all(abs(colMeans(obj$residuals$high)) < tol))

  Qlow <- qr.Q(qr(obj$pc_scores_raw$low))
  orth <- crossprod(Qlow, obj$residuals$high)
  expect_lt(
    max(abs(orth)),
    tol,
    "High-tier residuals should be orthogonal to the kept low-tier PCs (training data)"
  )
})

test_that("predict() on training data reproduces training residuals", {
  expect_true(!is.null(colnames(feature_list$low)))
  expect_true(!is.null(colnames(feature_list$high)))

  new_resid_on_train_data <- predict(obj, feature_list)
  expect_equal(new_resid_on_train_data$low, obj$residuals$low, tolerance = 1e-9)
  expect_equal(new_resid_on_train_data$high, obj$residuals$high, tolerance = 1e-9)
})

set.seed(43)
Xlow2 <- matrix(rnorm(n * p1), n, p1)
colnames(Xlow2) <- paste0("L", 1:p1)
signal2 <- Xlow2 %*% coef_matrix
Xhigh2 <- signal2 + matrix(rnorm(n * p2, sd = .3), n, p2)
colnames(Xhigh2) <- paste0("H", 1:p2)
newdata <- list(low = Xlow2, high = Xhigh2)

new_resid_on_new_data <- predict(obj, newdata)

test_that("dimensions are correct for new data prediction", {
  expect_named(new_resid_on_new_data, c("low", "high"))
  expect_equal(ncol(new_resid_on_new_data$low), 10)
  expect_equal(ncol(new_resid_on_new_data$high), 10)
  expect_equal(nrow(new_resid_on_new_data$high), n)
})

test_that("predict() on new data is batch-invariant", {
  idx <- seq(1L, 200L)
  from_full <- lapply(new_resid_on_new_data, function(m) m[idx, , drop = FALSE])
  from_sub <- predict(obj, list(low = Xlow2[idx, , drop = FALSE], high = Xhigh2[idx, , drop = FALSE]))
  expect_equal(from_sub$low, from_full$low, tolerance = 1e-9)
  expect_equal(from_sub$high, from_full$high, tolerance = 1e-9)
})

test_that("error when NA values are present", {
  feature_list_bad <- list(
    tier1 = matrix(c(1, NA, 3, 4), nrow = 2),
    tier2 = matrix(rnorm(4), nrow = 2)
  )
  expect_error(
    residualize_tiers(feature_list_bad, numpcs = 1),
    "Tier 'tier1' contains non-finite values \\(NA/Inf\\)\\."
  )
})

test_that("error when Inf values are present", {
  feature_list_bad <- list(
    tier1 = matrix(c(1, Inf, 3, 4), nrow = 2),
    tier2 = matrix(rnorm(4), nrow = 2)
  )
  expect_error(
    residualize_tiers(feature_list_bad, numpcs = 1),
    "Tier 'tier1' contains non-finite values \\(NA/Inf\\)\\."
  )
})

test_that("feature_list row mismatches trigger an error", {
  m1 <- matrix(rnorm(10), nrow = 5)
  m2 <- matrix(rnorm(8), nrow = 4)
  fl_bad <- list(first = m1, second = m2)
  expect_error(residualize_tiers(fl_bad), "same number of rows")
})
