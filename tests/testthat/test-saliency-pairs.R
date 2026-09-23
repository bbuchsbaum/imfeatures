saliency_pair_fixture <- function(cpp = TRUE, reference = FALSE, r_geometry = FALSE) {
  a <- matrix(0, 3, 2); a[1, 1] <- 1; a[3, 1] <- 2
  theta <- matrix(1L, 3, 2); theta[3, 1] <- 2L
  t <- matrix(0, 3, 2); t[1, 1] <- 1.5; t[3, 1] <- .5
  fun <- if (cpp) saliency_count_cpp else .saliency_count_r
  suppressWarnings(fun(theta, a, t, t, a > 0, 4L, 5L, 4L, TRUE, reference, r_geometry))
}

test_that("compiled pair counts agree with hand-enumerated ordered pairs", {
  both <- saliency_pair_fixture()
  expected <- array(0, c(5, 4, 4))
  expected[1, 2, 1] <- 2.25 # reference A, endpoint A
  expected[3, 2, 2] <- 1.5  # reference A, endpoint B
  expected[3, 1, 4] <- 1.5  # reference B, endpoint A
  expected[1, 3, 1] <- 1    # reference B, endpoint B
  expect_equal(both$counts, expected, tolerance = 1e-12)
  expect_equal(both$first_order, -(0.6*log2(.6) + .4*log2(.4)), tolerance = 1e-12)
  expect_equal(sum(both$squares), 10.5625, tolerance = 1e-12)
  expect_equal(sum(both$pairs), 4)
  reference <- saliency_pair_fixture(reference = TRUE)
  expected[1, 2, 1] <- 1.5; expected[3, 2, 2] <- 3
  expected[3, 1, 4] <- 1; expected[1, 3, 1] <- 2
  expect_equal(reference$counts, expected, tolerance = 1e-12)
  expect_equal(sum(reference$squares), 16.25, tolerance = 1e-12)
  for (geometry in c(FALSE, TRUE)) for (ref in c(FALSE, TRUE)) {
    expect_equal(saliency_pair_fixture(TRUE, ref, geometry),
                 saliency_pair_fixture(FALSE, ref, geometry), tolerance = 1e-12)
  }
})

test_that("first order known histograms and zero mass have native parity", {
  theta <- matrix(c(1L, 2L), 2)
  for (a in list(c(1, 0), c(1, 1), c(3, 1), c(0, 0))) {
    expected <- if (!sum(a)) NA_real_ else if (a[2] == 0) 0 else if (a[1] == 1) 1 else 0.8112781244591328
    for (cpp in c(TRUE, FALSE)) {
      fun <- if (cpp) saliency_count_cpp else .saliency_count_r
      value <- suppressWarnings(fun(theta, matrix(a, 2), matrix(1, 2), matrix(1, 2),
        matrix(TRUE, 2), 2L, 1L, 1L, FALSE, FALSE)$first_order)
      expect_equal(value, expected, tolerance = 1e-12)
    }
  }
  expect_error(saliency_count_cpp(theta, matrix(-1, 2), matrix(1, 2), matrix(1, 2),
    matrix(TRUE, 2), 2L, 1L, 1L, FALSE, FALSE), "Invalid response")
})

test_that("both new backends use the native legacy validity boundary", {
  counts <- array(c(0, .999995, 1, 1.00001), c(4, 1, 1))
  stats <- function(cpp, policy) .saliency_cell_stats(counts, matrix(counts^2, 4, 1),
    matrix(as.numeric(counts > 0), 4, 1), 0, cpp, policy, 0, 0)
  expect_identical(stats(FALSE, "legacy")$valid, c(FALSE, FALSE, FALSE, TRUE))
  expect_identical(stats(TRUE, "legacy")$valid, c(FALSE, FALSE, FALSE, TRUE))
  expect_identical(stats(FALSE, "support")$valid, c(FALSE, TRUE, TRUE, TRUE))
  expect_equal(stats(FALSE, "support"), stats(TRUE, "support"), tolerance = 1e-12)
  expect_true(is.na(stats(TRUE, "support")$entropy[1]))
})

test_that("batched controls preserve independently checked single-map counts", {
  a <- matrix(c(1, 0, 2, 4, 0, 3), 3, 2)
  theta <- matrix(c(1L, 2L, 2L, 4L, 3L, 1L), 3, 2)
  ts <- list(matrix(1, 3, 2), matrix(c(2, 0, 1, .5, 0, 0), 3, 2), matrix(0, 3, 2))
  for (ref in c(TRUE, FALSE)) {
    native <- saliency_count_batch_cpp(theta, a, ts, ts, a > 0, 4L, 5L, 4L, TRUE, ref)
    slow <- suppressWarnings(.saliency_count_batch_r(theta, a, ts, ts, a > 0, 4L, 5L, 4L, TRUE, ref))
    for (k in seq_along(ts)) {
      single <- saliency_count_cpp(theta, a, ts[[k]], ts[[k]], a > 0, 4L, 5L, 4L, TRUE, ref)
      expect_identical(native[[k]], single)
      expect_equal(native[[k]], slow[[k]], tolerance = 1e-12)
    }
  }
})

test_that("concentrated saliency can increase entropy and effective pair support", {
  a <- matrix(c(3, 1), 2)
  theta <- matrix(c(1L, 2L), 2)
  ts <- list(matrix(1, 2), matrix(c(.5, 1.5), 2))
  counted <- saliency_count_batch_cpp(theta, a, ts, ts, a > 0, 3L, 1L, 1L, TRUE, FALSE)
  stats <- lapply(counted, function(x) .saliency_cell_stats(x$counts, x$squares, x$pairs,
                                                           0:2, TRUE, "support", 0, 0))
  # Four ordered pairs: baseline relative-orientation counts (10,3,3),
  # versus (4.5,2.25,2.25) after saliency counterbalances the response strengths.
  expect_equal(stats[[1]]$entropy, 1.329434002924965, tolerance = 1e-12)
  expect_equal(stats[[2]]$entropy, 1.5, tolerance = 1e-12)
  expect_equal(stats[[1]]$effective_pairs, 2.56, tolerance = 1e-12)
  expect_equal(stats[[2]]$effective_pairs, 4, tolerance = 1e-12)
})

test_that("effective support is not a categorical entropy bound", {
  p <- (1 + sqrt(161))/24
  weights <- c(p, rep((1-p)/23, 23))
  result <- .saliency_cell_stats(array(weights, c(1, 1, 24)), matrix(sum(weights^2)),
    matrix(24), seq(0, 2*pi, length.out = 25)[-25], TRUE, "support", 0, 0)
  expect_equal(result$effective_pairs, 3, tolerance = 1e-12)
  expect_equal(result$entropy, 2.929184265652, tolerance = 1e-12)
  expect_gt(result$entropy, log2(result$effective_pairs))
  expect_equal(result$occupied_orientations, 24)
})

test_that("cell and range support use individual pair squares", {
  a <- matrix(c(1, 2), 2)
  theta <- matrix(1L, 2)
  for (cpp in c(TRUE, FALSE)) {
    fun <- if (cpp) saliency_count_cpp else .saliency_count_r
    counted <- fun(theta, a, matrix(1, 2), matrix(c(1.5, .5), 2), matrix(TRUE, 2),
                   1L, 1L, 1L, TRUE, FALSE, FALSE)
    result <- .saliency_pair_summary(counted, 0, list(c(1L, 1L), c(2L, 3L)),
                                    cpp, "support", 0, 0)
    expect_equal(result$cells$mass, 6.25)
    expect_equal(result$cells$squared_mass, 10.5625)
    expect_equal(result$cells$effective_pairs, 6.25^2/10.5625, tolerance = 1e-12)
    expect_equal(result$cells$unweighted_effective_pairs, 81/25, tolerance = 1e-12)
    expect_equal(result$ranges$effective_pairs[1], 6.25^2/10.5625, tolerance = 1e-12)
    expect_equal(result$ranges$valid_cell_coverage[1], 1)
    expect_equal(result$ranges$retained_mass_fraction[1], 1)
    expect_true(is.na(result$ranges$entropy[2]))
    expect_true(is.na(result$ranges$effective_pairs[2]))
    strict <- .saliency_pair_summary(counted, 0, list(c(1L, 1L)), cpp, "support", 0, 5)
    expect_false(strict$cells$valid)
    expect_equal(strict$ranges$retained_mass_fraction, 0)
    expect_equal(strict$ranges$valid_cell_coverage, 0)
  }
})

test_that("selection, ties, normalization and missing edge support are explicit", {
  a <- matrix(c(10, 9, 1, rep(0, 17)), 4)
  q <- matrix(c(0, 0, 1, rep(0, 17)), 4)
  expect_identical(which(.saliency_selection(a, q, "image")), 1:2)
  expect_true(.saliency_selection(a, q, "saliency")[3])
  a[3] <- 9
  expect_identical(which(.saliency_selection(a, q, "image")), 1:3)
  selected <- a > 0
  weights <- .saliency_edge_weights(q, selected)
  expect_equal(weights$weights[selected], c(0, 0, 3))
  expect_equal(weights, .saliency_edge_weights(q * 1e200, selected), tolerance = 1e-12)
  expect_equal(.saliency_edge_weights(matrix(7, 4, 5), selected)$weights[selected], rep(1, 3))
  expect_identical(.saliency_edge_weights(q, !selected)$status, "zero_edge_saliency")
  expect_identical(.saliency_edge_weights(q, matrix(FALSE, 4, 5))$status, "no_edges")
  large <- matrix(seq_len(10001), 10001, 1)
  selected_large <- .saliency_selection(large, large * 0 + 1, "image")
  expect_equal(sum(selected_large), 10000)
  expect_false(selected_large[1])
  large[1] <- 2
  expect_equal(sum(.saliency_selection(large, large * 0 + 1, "image")), 10001)
})
