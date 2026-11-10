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

test_that("grid bins are contiguous, cover every position and match cut()", {
  for (n in c(1, 2, 5, 7, 14, 15, 28, 56, 112, 224)) {
    for (b in seq_len(min(n, 8))) {
      idx <- imfeatures:::.bin_index(n, b)
      expect_length(idx, n)
      expect_false(is.unsorted(idx))
      expect_setequal(idx, seq_len(b))
      if (b >= 2) {
        expect_identical(
          idx,
          as.integer(cut(seq_len(n), breaks = b, labels = FALSE, include.lowest = TRUE))
        )
      }
    }
  }
  expect_equal(as.integer(table(imfeatures:::.bin_index(14, 4))), c(4, 3, 3, 4))
})

test_that("grid pooling accepts a single bin along either axis", {
  a <- make_nhwc(n = 2, h = 4, w = 6, c = 2)
  gap <- pool_activations(global_average_pool(), a)
  expect_equal(pool_activations(adaptive_grid_pool(1, 1), a), gap)

  strips <- pool_activations(adaptive_grid_pool(1, 3), a)
  expect_equal(dim(strips), c(2, 6))
  expect_equal(strips[1, 1], mean(a[1, , 1:2, 1]))
  expect_equal(strips[2, 6], mean(a[2, , 5:6, 2]))

  rows <- pool_activations(adaptive_grid_pool(2, 1), a)
  expect_equal(rows[1, 2], mean(a[1, 3:4, , 1]))
})

test_that("grid cell means reproduce GAP when the map divides evenly", {
  set.seed(11)
  a <- array(rnorm(2 * 6 * 6 * 2), dim = c(2, 6, 6, 2))
  grid <- pool_activations(adaptive_grid_pool(3, 2), a)
  gap <- pool_activations(global_average_pool(), a)
  expect_equal(rowMeans(grid[, 1:6]), gap[, 1])
  expect_equal(rowMeans(grid[, 7:12]), gap[, 2])
})

test_that("grid and DCT poolers reject sizes larger than the map", {
  a <- make_nhwc(h = 4, w = 4)
  expect_error(pool_activations(adaptive_grid_pool(5, 2), a), "exceeds spatial size")
  expect_error(pool_activations(dct_pool(2, 5), a), "exceed spatial size")
})

test_that("pooled columns follow pooler_feature_names order", {
  set.seed(3)
  a <- array(rnorm(2 * 6 * 4 * 3), dim = c(2, 6, 4, 3))

  grid <- adaptive_grid_pool(3, 2)
  out <- pool_activations(grid, a)
  nms <- pooler_feature_names(grid, "s", 3)
  expect_length(nms, ncol(out))
  expect_equal(out[2, which(nms == "s::channel_002::grid_y3_x1")], mean(a[2, 5:6, 1:2, 2]))
  expect_equal(out[1, which(nms == "s::channel_003::grid_y1_x2")], mean(a[1, 1:2, 3:4, 3]))

  ident <- pool_activations(identity_pool(), a)
  nms <- pooler_feature_names(identity_pool(), "s", 3, spatial = c(6, 4))
  expect_length(nms, ncol(ident))
  expect_equal(ident[2, which(nms == "s::channel_003::y5_x2")], a[2, 5, 2, 3])
  expect_equal(ident[1, which(nms == "s::channel_001::y1_x4")], a[1, 1, 4, 1])

  rad <- radial_pool()
  out <- pool_activations(rad, array(rnorm(2 * 8 * 8 * 3), dim = c(2, 8, 8, 3)))
  nms <- pooler_feature_names(rad, "s", 3)
  expect_length(nms, ncol(out))
  expect_equal(nms[4:6], paste0("s::channel_002::radial_r", 1:3))
})

test_that("DCT contrast columns have the documented sign, order and scale", {
  # Active on the left half only: positive left-right contrast, no
  # top-bottom contrast.
  a <- array(0, dim = c(1, 6, 6, 1))
  a[1, , 1:3, 1] <- 1
  pooler <- dct_pool(2, 2)
  out <- pool_activations(pooler, a)
  colnames(out) <- pooler_feature_names(pooler, "s", 1)
  expect_equal(unname(out[1, "s::channel_001::dct_y0_x0"]), 0.5)
  expect_gt(out[1, "s::channel_001::dct_y0_x1"], 0)
  expect_equal(unname(out[1, "s::channel_001::dct_y1_x0"]), 0, tolerance = 1e-12)

  # zigzag and raster keep the same modes in a different column order
  zig <- pool_activations(dct_pool(3, 3), a)
  ras <- pool_activations(dct_pool(3, 3, order = "raster"), a)
  zn <- pooler_feature_names(dct_pool(3, 3), "s", 1)
  rn <- pooler_feature_names(dct_pool(3, 3, order = "raster"), "s", 1)
  expect_setequal(zn, rn)
  expect_equal(zig[1, match(rn, zn)], ras[1, ])
  expect_equal(
    sub(".*::dct_", "", zn[1:6]),
    c("y0_x0", "y0_x1", "y1_x0", "y2_x0", "y1_x1", "y0_x2")
  )

  # A unit cosine along width gives sqrt(H * W / 2) for an orthonormal mode.
  for (n in c(8, 14)) {
    cosmap <- array(0, dim = c(1, n, n, 1))
    cosmap[1, , , 1] <- outer(rep(1, n), cos(pi * (2 * seq_len(n) - 1) / (2 * n)))
    expect_equal(pool_activations(dct_pool(1, 2), cosmap)[1, 2], sqrt(n * n / 2))
  }
})

test_that("radial rings average the positions at each normalized radius", {
  a <- array(0, dim = c(1, 14, 14, 1))
  a[1, , , 1] <- seq_len(14 * 14)
  yy <- row(matrix(0, 14, 14))
  xx <- col(matrix(0, 14, 14))
  r <- sqrt((yy - 7.5)^2 + (xx - 7.5)^2)
  r <- r / max(r)
  out <- pool_activations(radial_pool(), a)
  expect_equal(out[1, 1], mean(a[1, , , 1][r < 0.25]))
  expect_equal(out[1, 2], mean(a[1, , , 1][r >= 0.25 & r < 0.5]))
  expect_equal(out[1, 3], mean(a[1, , , 1][r >= 0.5]))
  expect_equal(c(sum(r < 0.25), sum(r >= 0.25 & r < 0.5), sum(r >= 0.5)), c(16, 52, 128))

  # Positions beyond the last break are left out.
  centre <- pool_activations(radial_pool(c(0, 0.5)), a)
  expect_equal(centre[1, 1], mean(a[1, , , 1][r <= 0.5]))
})

test_that("radial_pool validates breaks and warns about empty rings", {
  expect_error(radial_pool(c(0, 0.5, 2)))
  expect_error(radial_pool(c(-0.1, 1)))
  expect_error(radial_pool(c(0, 0.5, 0.5, 1)), "strictly increasing")
  expect_error(radial_pool(1))

  tiny <- array(1:6, dim = c(2, 1, 1, 3))
  expect_warning(
    out <- pool_activations(radial_pool(), tiny),
    "hold no positions on a 1 x 1 map"
  )
  expect_equal(out[, c(1, 4, 7)], matrix(1:6, 2, 3), ignore_attr = TRUE)
  expect_true(all(out[, -c(1, 4, 7)] == 0))

  # The centre of an even-sized map lies between positions, so a 4 x 4 map
  # has nothing within 0.25 of it.
  expect_warning(
    pool_activations(radial_pool(), make_nhwc(h = 4, w = 4)),
    "Radial ring\\(s\\) 1 hold no positions on a 4 x 4 map"
  )
  expect_silent(pool_activations(radial_pool(), make_nhwc(h = 7, w = 7)))
  expect_silent(pool_activations(radial_pool(), make_nhwc(h = 14, w = 14)))
})

test_that("dense matrices pass through every pooler unchanged", {
  m <- matrix(as.numeric(1:10), 2, 5)
  poolers <- list(
    global_average_pool(), global_max_pool(), adaptive_grid_pool(2, 2),
    dct_pool(2, 2), radial_pool(), identity_pool()
  )
  for (p in poolers) {
    expect_identical(pool_activations(p, m), m)
  }
  expect_error(
    pool_activations(global_average_pool(), array(0, dim = c(2, 2, 2))),
    "matrix or a 4-D array"
  )
})

test_that("a single image keeps its row dimension", {
  a <- make_nhwc(n = 1, h = 8, w = 8, c = 3)
  expect_equal(dim(pool_activations(global_average_pool(), a)), c(1, 3))
  expect_equal(dim(pool_activations(global_max_pool(), a)), c(1, 3))
  expect_equal(dim(pool_activations(adaptive_grid_pool(2, 2), a)), c(1, 12))
  expect_equal(dim(pool_activations(dct_pool(2, 2), a)), c(1, 12))
  expect_equal(dim(pool_activations(radial_pool(), a)), c(1, 9))
  expect_equal(dim(pool_activations(identity_pool(), a)), c(1, 192))
})

test_that("every pooler gives the same result for NCHW and NHWC input", {
  set.seed(5)
  nhwc <- array(rnorm(2 * 10 * 8 * 3), dim = c(2, 10, 8, 3))
  nchw <- aperm(nhwc, c(1, 4, 2, 3))
  poolers <- list(
    global_max_pool(), adaptive_grid_pool(3, 2), dct_pool(3, 2),
    radial_pool(), identity_pool()
  )
  for (p in poolers) {
    expect_equal(
      pool_activations(p, nchw, layout = "nchw"),
      pool_activations(p, nhwc, layout = "nhwc")
    )
  }
})
