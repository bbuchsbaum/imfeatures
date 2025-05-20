library(testthat)
#library(imfeatures)

context("entropy")

test_that("entropy handles numeric vector", {
  x <- c(0.2, 0.3, 0.5)
  expect_equal(imfeatures:::entropy(x), -sum(x * log2(x)))
})

test_that("entropy removes NA and renormalizes", {
  x <- c(0.2, 0.3, NA, 0.5)
  expect_equal(imfeatures:::entropy(x), imfeatures:::entropy(c(0.2, 0.3, 0.5)))
})

test_that("entropy does not renormalize when sum is 1", {
  x <- c(0.2, 0.8)
  expect_equal(imfeatures:::entropy(x), -sum(x * log2(x)))
})

# Additional edge cases

test_that("entropy returns NA with warning for invalid inputs", {
  expect_warning(res1 <- imfeatures:::entropy(c(NA, NA)), "All values are NA")
  expect_true(is.na(res1))

  expect_warning(res2 <- imfeatures:::entropy(c(0, 0, 0)), "Sum of probabilities is zero")
  expect_true(is.na(res2))
})
