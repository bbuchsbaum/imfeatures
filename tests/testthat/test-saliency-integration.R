test_that("indicator weighting equals entropy of responses restricted to the region", {
  img <- outer(1:24, 1:25, function(x, y) sin(x*.7) + cos(y*.3))
  region <- matrix(FALSE, 24, 25); region[4:12, 6:19] <- TRUE
  filtered <- saliency_filter_cpp(img, 8L, 7L)
  hist <- vapply(1:8, function(b) sum(filtered$resp_val[region & filtered$resp_bin == b]), numeric(1))
  p <- hist[hist > 0] / sum(hist)
  expected <- -sum(p * log2(p))
  for (cpp in c(FALSE, TRUE)) {
    out <- compute_saliency_features(img, region * 1, "weights", pairwise = FALSE,
      filter_length = 7L, gabor_bins = 8L, use_cpp = cpp)
    expect_equal(out$features$first_order[1], expected, tolerance = 1e-12)
  }
})

test_that("saliency localizes low orientation diversity in a mixed image", {
  set.seed(943)
  img <- outer(1:80, 1:60, function(x, y) sin(x * .6))
  img[41:80, ] <- matrix(rnorm(40*60), 40, 60)
  left <- matrix(0, 80, 60); left[10:30, 10:50] <- 1
  right <- left[80:1, ]
  run <- function(q) compute_saliency_features(img, q, "weights", pairwise = FALSE,
    filter_length = 11L, gabor_bins = 8L, transform_controls = FALSE)$features$first_order[1]
  expect_lt(run(left), run(right) - .5)
})

test_that("public selection and reference weighting have backend parity", {
  set.seed(45)
  img <- matrix(rnorm(24*25), 24, 25)
  q <- matrix(0, 24, 25); q[4:12, 4:18] <- matrix(seq_len(9*15), 9, 15)
  run <- function(cpp, selection, weighting) compute_saliency_features(img, q, "weights",
    filter_length = 7L, gabor_bins = 8L, maxdiag = 30L, circ_bins = 8L,
    ranges = list(c(1L, 10L), c(11L, 20L), c(21L, 30L)), use_cpp = cpp,
    edge_selection = selection, pair_weighting = weighting, transform_controls = FALSE)
  for (selection in c("image", "saliency")) for (weighting in c("both", "reference")) {
    native <- run(TRUE, selection, weighting)
    slow <- run(FALSE, selection, weighting)
    expect_equal(native$features, slow$features, tolerance = 1e-10)
    expect_equal(native$ranges, slow$ranges, tolerance = 1e-10)
    expect_equal(native$cells, slow$cells, tolerance = 1e-10)
    expect_identical(native$selection, slow$selection)
    expect_equal(native$comparisons, slow$comparisons, tolerance = 1e-10)
    r <- native$ranges
    expect_equal(r$observed_minus_control,
      r$entropy[r$control == "observed"][r$range_id] - r$entropy)
    if (selection == "saliency") {
      expect_false(identical(native$selection$observed, native$selection$unweighted))
      filtered <- saliency_filter_cpp(img, 8L, 7L)
      score <- filtered$resp_val * q
      cutoff <- sort(as.vector(score), decreasing = TRUE)[ceiling(length(img)*.1)]
      expect_identical(native$selection$observed,
        which(filtered$resp_val > 0 & score >= cutoff, arr.ind = TRUE))
    }
  }
  symmetric <- run(TRUE, "saliency", "both")
  reference <- run(TRUE, "saliency", "reference")
  expect_false(isTRUE(all.equal(symmetric$cells$observed$mass, reference$cells$observed$mass)))
})

test_that("unavailable donor assignments do not borrow baseline entropy", {
  img <- outer(1:20, 1:21, function(x, y) sin(x) + cos(y))
  assignment <- saliency_donor_assignments("a", "alone", replicates = 1L)
  for (cpp in c(TRUE, FALSE)) {
    out <- compute_saliency_features(img, abs(img), "weights", image_id = "a",
      donor_assignments = assignment, use_cpp = cpp, filter_length = 5L,
      gabor_bins = 4L, maxdiag = 18L, circ_bins = 4L, ranges = list(c(1L, 18L)))
    donor <- out$features[out$features$control == "donor_1", ]
    expect_identical(donor$first_order_status, "no_valid_permutation")
    expect_true(is.na(donor$first_order))
    expect_true(all(out$cells$donor_1$mass == 0))
    expect_true(all(out$cells$donor_1$status == "no_valid_permutation"))
    expect_false(any(out$cells$donor_1$valid))
    expect_true(all(out$cells$unweighted$mass == out$cells$donor_1$unweighted_mass))
    expect_true(is.na(out$ranges$entropy[out$ranges$control == "donor_1"]))
  }
})

test_that("realistic default-bin analysis shares one counting group", {
  skip_on_cran()
  set.seed(952)
  img <- matrix(runif(150*150), 150, 150)
  q <- outer(1:150, 1:150, function(x, y) exp(-((x-45)^2+(y-90)^2)/600))
  out <- compute_saliency_features(img, q, "weights")
  expect_equal(out$metadata$counting_groups, 1L)
  expect_equal(nrow(out$selection$observed), 10000L)
  expect_equal(nrow(out$cells$observed), 500L * 48L)
  expect_true(all(is.finite(out$features$first_order[1:2])))
  observed <- out$ranges[out$ranges$control == "observed", ]
  expect_true(all(is.finite(observed$entropy[1:2])))
  # The compact hotspot has positive far-range mass but none above the legacy
  # cell cutoff. Availability must follow support, not image size alone.
  expect_true(is.na(observed$entropy[3]))
  expect_identical(observed$status[3], "insufficient_support")
  expect_gt(observed$mass[3], 0)
  expect_equal(observed$valid_cell_coverage[3], 0)
  expect_true(any(out$comparisons$status == "ok"))
})
