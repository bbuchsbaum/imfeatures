context("convolution boundary handling")

frame_depth <- function(m) {
  outer(pmin(seq_len(nrow(m)), rev(seq_len(nrow(m)))),
        pmin(seq_len(ncol(m)), rev(seq_len(ncol(m)))), pmin)
}

test_that("zero padding leaves frame responses past the 2-pixel border; reflect does not", {
  filt <- imfeatures:::saliency_filter_cpp
  flat <- matrix(0.5, 80, 70)
  zero <- filt(flat, 24L, 31L, "zero")$resp_val
  refl <- filt(flat, 24L, 31L, "reflect")$resp_val
  depth <- frame_depth(flat)
  expect_true(all(zero[depth <= 2] == 0))
  expect_gt(min(zero[depth == 3]), 0.1)
  expect_lt(max(abs(refl)), 1e-10)
})

test_that("reflection only changes responses near the frame", {
  set.seed(3)
  img <- matrix(runif(90 * 80), 90, 80)
  filt <- imfeatures:::saliency_filter_cpp
  zero <- filt(img, 24L, 31L, "zero")
  refl <- filt(img, 24L, 31L, "reflect")
  interior <- frame_depth(img) > 15
  expect_equal(zero$resp_val[interior], refl$resp_val[interior], tolerance = 1e-12)
  expect_identical(zero$resp_bin[interior], refl$resp_bin[interior])
})

test_that("reflection is defined for images smaller than the kernel", {
  set.seed(4)
  img <- matrix(runif(9 * 12), 9, 12)
  out <- imfeatures:::saliency_filter_cpp(img, 8L, 31L, "reflect")
  expect_equal(dim(out$resp_val), dim(img))
  expect_true(all(is.finite(out$resp_val)))
})

test_that("frame edges no longer dominate an object on a plain background", {
  img <- matrix(1, 160, 150)
  yy <- row(img); xx <- col(img)
  disc <- (yy - 80)^2 + (xx - 75)^2 < 45^2
  img[disc] <- 0.5 + 0.4 * sin(yy[disc] / 3) * cos(xx[disc] / 5)
  filt <- imfeatures:::saliency_filter_cpp
  share <- function(boundary) {
    a <- filt(img, 24L, 31L, boundary)$resp_val
    top <- a >= sort(a, decreasing = TRUE)[2000]
    mean(frame_depth(img)[top] <= 6)
  }
  expect_gt(share("zero"), 0.1)
  expect_equal(share("reflect"), 0)
})

test_that("default boundary reproduces earlier results and arguments are validated", {
  set.seed(5)
  img <- matrix(runif(60 * 50), 60, 50)
  expect_identical(compute_edge_entropy(img), compute_edge_entropy(img, boundary = "zero"))
  refl <- compute_edge_entropy(img, boundary = "reflect")
  expect_false(isTRUE(all.equal(refl$entropy, compute_edge_entropy(img)$entropy)))
  expect_error(compute_edge_entropy(img, boundary = "wrap"))
  expect_error(imfeatures:::edge_entropy_cpp(img, boundary = "wrap"), "boundary")
})

test_that("R and C++ backends agree with reflection padding", {
  skip_if_not_installed("imager")
  set.seed(6)
  img <- matrix(runif(60 * 55), 60, 55)
  bank <- imfeatures:::filter_bank(24L, 31L)
  r <- imfeatures:::run_filterbank(list(image_raw = img, image_size = dim(img)), bank,
    boundary = "reflect")
  cpp <- imfeatures:::saliency_filter_cpp(img, 24L, 31L, "reflect")
  expect_equal(r$resp_val, cpp$resp_val, tolerance = 1e-8)
  expect_identical(r$resp_bin, cpp$resp_bin)
  res_r <- compute_edge_entropy(img, use_cpp = FALSE, boundary = "reflect")
  res_c <- compute_edge_entropy(img, use_cpp = TRUE, boundary = "reflect")
  expect_equal(res_c$entropy, res_r$entropy, tolerance = 5e-4)
  expect_equal(res_c$complex_before, res_r$complex_before, tolerance = 5e-4)
})
