library(testthat)
library(imfeatures)

context("do_counting handles small images")

test_that("do_counting works on very small images", {
  img <- matrix(runif(9), nrow = 3)
  fimg <- list(
    file = "matrix",
    img = NULL,
    image_raw = img,
    image_size = dim(img)
  )
  class(fimg) <- c("filtered_image", "list")
  fbank <- filter_bank(4, 3)
  fres  <- run_filterbank(fimg, fbank)
  expect_silent(res <- do_counting(fres))
  expect_true(is.list(res))
  expect_true(all(c("counts", "complex_before") %in% names(res)))
})
