# -------------------------------------------------------------------------
#  UNIT TEST  — residualize_tiers()
# -------------------------------------------------------------------------
library(testthat)
# Ensure imfeatures package is loaded, typically handled by devtools::test() or test_dir()
# library(imfeatures) # Or use imfeatures::residualize_tiers if not attaching

context("residualize_tiers: PCA and hierarchical residualization")

set.seed(42)
tol <- 1e-8 # Define a general tolerance

## ------------------------------------------------------
## 1.  Make synthetic tiered data with known collinearity
## ------------------------------------------------------
n   <- 500                       # samples (frames)
p1  <- 30                        # features in Low tier
p2  <- 40                        # features in High tier

# Low-tier features: iid Gaussian
Xlow <- matrix(rnorm(n * p1), n, p1)
colnames(Xlow) <- paste0("L", 1:p1) # Add colnames for robust predict test

# High-tier raw features: 70 % collinear with Low + 30 % unique noise
# Use a fixed coefficient matrix for reproducibility of signal component
set.seed(123) # For coefficients
coef_matrix <- matrix(rnorm(p1 * p2, sd = .7), p1, p2)
set.seed(42)  # Reset main seed

signal      <- Xlow %*% coef_matrix
noise       <- matrix(rnorm(n * p2, sd = .3), n, p2)
Xhigh       <- signal + noise
colnames(Xhigh) <- paste0("H", 1:p2) # Add colnames

feature_list <- list(low = Xlow, high = Xhigh)

## ------------------------------------------------------
## 2.  Residualize
## ------------------------------------------------------
# source("residualize_tiers.R")   # Not needed if package is loaded
obj <- residualize_tiers(feature_list, numpcs = 10, pca_method = "stats")

## ------------------------------------------------------
## 3.  Core checks
## ------------------------------------------------------
test_that("dimensions are correct", {
  expect_s3_class(obj, "residualized_tiers")
  expect_named(obj$residuals, c("low", "high"))
  expect_equal(ncol(obj$residuals$low), 10)
  expect_equal(ncol(obj$residuals$high), 10)
  expect_equal(nrow(obj$residuals$high), n)
})

test_that("low and residualized high are orthogonal (training data)", {
  # Residuals from the object are already the ones we want to check
  # obj$residuals$low are the PCs of Xlow (as it's the first tier)
  # obj$residuals$high are the residuals of Xhigh PCs after projecting out obj$pc_scores$low space
  
  # Check that residuals are reasonably centered (PCA should handle this)
  expect_true(all(abs(colMeans(obj$residuals$low)) < tol))
  expect_true(all(abs(colMeans(obj$residuals$high)) < tol))

  # Orthogonality check: residuals$high should be orthogonal to the *kept* pc_scores$low
  # obj$pc_scores$low are the scores used for residualizing the high tier.
  Qlow <- qr.Q(qr(obj$pc_scores$low))          # Basis of kept low PCs
  orth  <- crossprod(Qlow, obj$residuals$high) # Should be (k_low × k_high) ~ 0
  expect_lt(max(abs(orth)), tol, "High-tier residuals should be orthogonal to the kept low-tier PCs (training data)")
})

test_that("predict() on training data reproduces training residuals", {
  # Ensure colnames are present in feature_list for robust test of predict
  expect_true(!is.null(colnames(feature_list$low)))
  expect_true(!is.null(colnames(feature_list$high)))
  
  new_resid_on_train_data <- predict(obj, feature_list)
  expect_equal(new_resid_on_train_data$low,  obj$residuals$low,  tolerance = 1e-9)
  expect_equal(new_resid_on_train_data$high, obj$residuals$high, tolerance = 1e-9)
})

## ------------------------------------------------------
## 4.  New data with same generative structure
## ------------------------------------------------------
# Create new data. Use the same coefficient matrix for signal component for consistency.
set.seed(43) # Different seed for new Xlow2 and noise2
Xlow2  <- matrix(rnorm(n * p1), n, p1); colnames(Xlow2) <- paste0("L", 1:p1)
signal2 <- Xlow2 %*% coef_matrix # Same projection from Xlow2
Xhigh2  <- signal2 + matrix(rnorm(n * p2, sd = .3), n, p2); colnames(Xhigh2) <- paste0("H", 1:p2)
newdata <- list(low = Xlow2, high = Xhigh2)

new_resid_on_new_data <- predict(obj, newdata)

test_that("dimensions are correct for new data prediction", {
  expect_named(new_resid_on_new_data, c("low", "high"))
  expect_equal(ncol(new_resid_on_new_data$low), 10)
  expect_equal(ncol(new_resid_on_new_data$high), 10)
  expect_equal(nrow(new_resid_on_new_data$high), n)
})

test_that("low and residualized high are orthogonal (new data)", {
  # For new data, new_resid_on_new_data$low are PCs of Xlow2.
  # new_resid_on_new_data$high are residuals of Xhigh2 PCs after projecting out the space
  # spanned by new_resid_on_new_data$low (which are PCs of Xlow2).
  
  # Note: PC scores on new data (new_resid_on_new_data$low and the unresidualized new_resid_on_new_data$high)
  # are not guaranteed to have zero mean if the new data's mean differs from the training data's mean.
  # The predict method correctly applies the PCA transformation (which includes centering with training data mean).
  # Thus, we don't check for zero means here, but focus on the orthogonality.
  
  # Orthogonality check: new_resid_on_new_data$high should be orthogonal to new_resid_on_new_data$low
  Qlow_new <- qr.Q(qr(new_resid_on_new_data$low))
  orth2    <- crossprod(Qlow_new, new_resid_on_new_data$high)
  expect_lt(max(abs(orth2)), tol, "High-tier residuals should be orthogonal to the kept low-tier PCs (new data)")
})

# cat("All residualize_tiers tests passed ✔\n") # This cat() is more for interactive script, testthat has its own reporting 