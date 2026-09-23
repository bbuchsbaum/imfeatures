test_that("map representation and zero mass are explicit", {
  expect_error(saliency_location_entropy(matrix(1, 2, 2)), "explicitly")
  expect_error(saliency_location_entropy(matrix(-1, 2, 2), "weights"), "Negative")
  expect_error(saliency_location_entropy(matrix(NA_real_, 2, 2), "weights"), "finite")
  expect_error(saliency_location_entropy(matrix(Inf, 2, 2), "log_density"), "finite")
  expect_error(saliency_location_entropy(matrix(-1, 2, 2), "log_density", "rectify"), "only")
  for (type in c("weights", "log_density")) {
    value <- if (type == "weights") 0 else -Inf
    result <- saliency_location_entropy(matrix(value, 2, 2), type)
    expect_identical(result$status, "zero_map_mass")
    expect_true(all(is.na(result[1:3])))
  }
  x <- matrix(c(-2, -1, 0, 2), 2)
  expect_equal(saliency_location_entropy(x, "weights", "rectify")$H_location, 0)
  expect_equal(saliency_location_entropy(x, "weights", "shift"),
               saliency_location_entropy(x + 2, "weights"))
  expect_equal(saliency_location_entropy(x * 1e200, "weights", "shift"),
               saliency_location_entropy(x, "weights", "shift"))
})

test_that("downsampling averages narrow probability support without aliasing", {
  q <- matrix(0, 4, 4); q[2, 2] <- .25; q[4, 4] <- .75
  expected <- matrix(c(.25, 0, 0, .75), 2, 2)
  expect_equal(.saliency_resize(q, c(2L, 2L)), expected, tolerance = 1e-12)
  expect_equal(.saliency_resize(q*0, c(2L, 2L)), expected*0)
  expect_equal(sum(.saliency_resize(q, c(2L, 8L))), 1)
  expect_identical(.saliency_resize_method(c(4, 4), c(2, 8)), "area_then_linear")
})

test_that("location concentration has independently known limits and invariants", {
  uniform <- saliency_location_entropy(matrix(1, 2, 4), "weights")
  expect_equal(uniform$H_location, 3)
  expect_equal(uniform$KL_uniform, 0)
  expect_equal(uniform$effective_fraction, 1)
  point <- matrix(0, 2, 4); point[1, 1] <- 1
  result <- saliency_location_entropy(point, "weights")
  expect_equal(result$H_location, 0)
  expect_equal(result$KL_uniform, 3)
  expect_equal(result$effective_fraction, 1/8)
  x <- matrix(c(0, 1, 2, 3), 2)
  ref <- saliency_location_entropy(x, "weights")
  expect_equal(ref, saliency_location_entropy(x * 1e200, "weights"), tolerance = 1e-12)
  expect_equal(ref, saliency_location_entropy(log(x) - 1000, "log_density"), tolerance = 1e-12)
  refined <- kronecker(x, matrix(1/4, 2, 2))
  out <- saliency_location_entropy(refined, "weights")
  expect_equal(out$H_location, ref$H_location + 2, tolerance = 1e-12)
  expect_equal(out$KL_uniform, ref$KL_uniform, tolerance = 1e-12)
  expect_equal(out$effective_fraction, ref$effective_fraction, tolerance = 1e-12)
})

test_that("multiscale blocks conserve mass and account for boundary area", {
  uniform <- saliency_multiscale_entropy(matrix(1, 3, 3), "weights", scales = 2,
                                        units = "pixels", origins = matrix(c(0, 0), 1))
  # Four blocks with areas 4, 2, 2, 1; uniform mass follows those areas.
  expected <- -(4/9*log2(4/9) + 2*2/9*log2(2/9) + 1/9*log2(1/9))
  expect_equal(uniform$curve$H_location, expected, tolerance = 1e-12)
  expect_equal(uniform$curve$KL_uniform, 0, tolerance = 1e-12)
  expect_equal(uniform$curve$blocks, 4)
  shifted <- saliency_multiscale_entropy(matrix(1, 3, 3), "weights", scales = 2, units = "pixels")
  expect_equal(shifted$curve$KL_uniform, rep(0, 4), tolerance = 1e-12)
  a <- matrix(0, 4, 4); a[1, 1:2] <- 1
  b <- matrix(0, 4, 4); b[1, c(1, 3)] <- 1
  measure <- function(x) saliency_multiscale_entropy(x, "weights", scales = c(1, 2),
                          units = "pixels", origins = matrix(c(0, 0), 1))$curve$H_location
  expect_equal(measure(a), c(1, 0))
  expect_equal(measure(b), c(1, 1))
  expect_error(saliency_multiscale_entropy(a, "weights", units = "degrees"), "pixels_per_degree")
  expect_equal(saliency_multiscale_entropy(a, "weights", scales = 1, units = "degrees",
    pixels_per_degree = 2)$curve$H_location,
    saliency_multiscale_entropy(a, "weights", scales = .5, units = "relative")$curve$H_location,
    tolerance = 1e-12)
  expect_equal(saliency_multiscale_entropy(a, "weights", scales = 1, units = "degrees",
    pixels_per_degree = 2)$curve$block_pixels, rep(2, 4))
  expect_true(all(is.na(saliency_multiscale_entropy(matrix(0, 2, 2), "weights")$curve$H_location)))
})
