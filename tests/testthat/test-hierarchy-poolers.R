library(testthat)

context("hierarchy poolers")

make_nhwc <- function(n = 2, h = 4, w = 4, c = 2) {
  a <- array(0, dim = c(n, h, w, c))
  for (i in seq_len(n)) {
    for (ch in seq_len(c)) {
      a[i, , , ch] <- matrix(seq_len(h * w) + 10 * i + 100 * ch, h, w)
    }
  }
  a
}

test_that("as_nhwc converts NCHW to NHWC", {
  nchw <- array(seq_len(2 * 3 * 4 * 5), dim = c(2, 3, 4, 5))
  nhwc <- as_nhwc(nchw, layout = "nchw")
  expect_equal(dim(nhwc), c(2, 4, 5, 3))
  expect_equal(nhwc[1, 2, 3, 1], nchw[1, 1, 2, 3])
  expect_identical(as_nhwc(nchw, layout = "nhwc"), nchw)
})

test_that("global average pooling matches channel means", {
  a <- make_nhwc()
  out <- pool_activations(global_average_pool(), a)
  expect_equal(dim(out), c(2, 2))
  expect_equal(out[1, 1], mean(a[1, , , 1]))
  expect_equal(out[2, 2], mean(a[2, , , 2]))
})

test_that("global max pooling matches channel maxima", {
  a <- make_nhwc()
  out <- pool_activations(global_max_pool(), a)
  expect_equal(out[1, 1], max(a[1, , , 1]))
})

test_that("DCT basis is orthonormal except the GAP-scaled DC mode", {
  basis <- imfeatures:::.dct_2d_basis(6, 6, 4, 4, "zigzag")
  gram <- tcrossprod(basis)
  # DC was rescaled to equal GAP, so it is not orthonormal with the rest.
  expect_equal(gram[-1, -1], diag(15), tolerance = 1e-10)
  dc <- matrix(basis[1, ], 6, 6)
  expect_true(all(abs(dc - dc[1, 1]) < 1e-12))
  expect_equal(dc[1, 1], 1 / 36, tolerance = 1e-12)
})

test_that("DCT DC coefficient equals global average", {
  a <- make_nhwc(n = 3, h = 6, w = 6, c = 2)
  dct <- pool_activations(dct_pool(2, 2), a)
  gap <- pool_activations(global_average_pool(), a)
  # first mode per channel is DC
  expect_equal(dct[, 1], gap[, 1], tolerance = 1e-10)
  expect_equal(dct[, 5], gap[, 2], tolerance = 1e-10)
})

test_that("zigzag starts at DC and then the lowest frequencies", {
  pairs <- imfeatures:::.dct_mode_pairs(3, 3, "zigzag")
  expect_equal(pairs$v[1], 0)
  expect_equal(pairs$u[1], 0)
  expect_equal(length(pairs$v), 9)
})

test_that("adaptive grid pooling averages rectangular bins", {
  a <- array(0, dim = c(1, 4, 4, 1))
  a[1, 1:2, 1:2, 1] <- 1
  a[1, 1:2, 3:4, 1] <- 2
  a[1, 3:4, 1:2, 1] <- 3
  a[1, 3:4, 3:4, 1] <- 4
  out <- pool_activations(adaptive_grid_pool(2, 2), a)
  expect_equal(as.numeric(out), c(1, 2, 3, 4), tolerance = 1e-12)
})

test_that("radial pooling uses three default rings", {
  a <- make_nhwc(n = 1, h = 8, w = 8, c = 1)
  out <- pool_activations(radial_pool(), a)
  expect_equal(ncol(out), 3)
  expect_true(all(is.finite(out)))
})

test_that("identity pool is a no-op on dense matrices", {
  m <- matrix(1:10, 2, 5)
  expect_equal(pool_activations(identity_pool(), m), m)
})

test_that("feature names encode stage, channel, and spatial mode", {
  nms <- pooler_feature_names(dct_pool(2, 2), "b2", 3)
  expect_equal(length(nms), 12)
  expect_true(all(grepl("^b2::channel_00[123]::dct_y[01]_x[01]$", nms)))
  expect_equal(nms[1], "b2::channel_001::dct_y0_x0")
})

test_that("NCHW input matches NHWC after conversion", {
  nhwc <- make_nhwc()
  nchw <- aperm(nhwc, c(1, 4, 2, 3))
  expect_equal(
    pool_activations(global_average_pool(), nchw, layout = "nchw"),
    pool_activations(global_average_pool(), nhwc, layout = "nhwc")
  )
})
