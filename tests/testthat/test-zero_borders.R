library(testthat)
#library(imfeatures)

context("zero_borders edge cases")

test_that("tiny matrices keep dimensions", {
  m <- matrix(1:4, nrow = 2)
  res <- imfeatures:::zero_borders(m, nlines = 2)
  expect_equal(dim(res), c(2, 2))

  res2 <- imfeatures:::zero_borders(m, nlines = 5)
  expect_equal(dim(res2), c(2, 2))
})
