library(testthat)
library(imfeatures)

context(".add_feature_dimnames helper function")

test_that("dimnames set for matrix", {
  mat <- matrix(1:4, nrow = 2)
  out <- imfeatures:::.add_feature_dimnames(mat, c("img1", "img2"))
  expect_equal(rownames(out), c("img1", "img2"))
})

test_that("dimnames set for array", {
  arr <- array(1:8, dim = c(2,2,2))
  out <- imfeatures:::.add_feature_dimnames(arr, c("img1", "img2"))
  expect_equal(dimnames(out)[[1]], c("img1", "img2"))
})
