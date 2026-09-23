test_that("common support uses identical cells and the distance hierarchy", {
  observed <- data.frame(distance_index = rep(1:2, 3), mass = 1,
    entropy = c(0, 2, 2, 4, 8, 6), valid = c(TRUE, TRUE, TRUE, FALSE, FALSE, TRUE))
  control <- observed
  control$entropy <- c(1, 5, 3, 7, 9, 8)
  control$valid <- c(TRUE, TRUE, FALSE, TRUE, TRUE, TRUE)
  out <- .saliency_common_summary(list(observed, control), c(1L, 2L), 0)
  # Shared cells: distance 1 has (0,1); distance 2 has (2,5) and (6,8).
  expect_equal(out$values, c(2, 3.75))
  expect_equal(diff(out$values), 1.75)
  expect_equal(out$coverage, .5)
  expect_equal(out$distance_coverage, 1)
  expect_equal(out$retained, c(.5, .5))
  rejected <- .saliency_common_summary(list(observed, control), c(1L, 2L), .6)
  expect_identical(rejected$status, "insufficient_common_support")
  expect_true(all(is.na(rejected$values)))
  control$valid <- !observed$valid
  empty <- .saliency_common_summary(list(observed, control), c(1L, 2L), 0)
  expect_identical(empty$status, "insufficient_common_support")
  expect_true(all(is.na(empty$values)))
})

test_that("family summaries use distinct controls and a family-wide mask", {
  img <- outer(1:20, 1:21, function(x, y) sin(x*.7) + cos(y*.3))
  q <- outer(1:20, 1:21, function(x, y) exp(-((x-7)^2+(y-9)^2)/25))
  assignments <- saliency_donor_assignments(c("a", "b"), rep("same", 2), replicates = 3)
  out <- compute_saliency_features(img, q, "weights", image_id = "a",
    donor_maps = list(b = q[20:1, ]), donor_assignments = assignments,
    filter_length = 5L, gabor_bins = 4L, maxdiag = 18L, circ_bins = 4L,
    ranges = list(c(1L, 18L)), centre_match = TRUE)
  donor <- out$comparisons[out$comparisons$family == "donor", ]
  expect_equal(donor$n_requested, c(3, 3))
  expect_equal(donor$n_controls, c(1, 1))
  expect_identical(donor$controls_used, c("donor_1", "donor_1"))
  expect_equal(donor$observed_minus_control_mean[1], out$features$first_order[1] -
    out$features$first_order[out$features$control == "donor_1"])
  expect_equal(donor$observed_minus_control_mean[2],
    out$ranges$observed_minus_control_common[out$ranges$control == "donor_1"])
  expect_equal(out$ranges$observed_minus_donor_mean[out$ranges$control == "observed"],
               donor$observed_minus_control_mean[2])
  ids <- c("observed", "flip_axis1", "flip_axis2", "rotate180")
  joint_mask <- Reduce(`&`, lapply(out$cells[ids], function(x) x$valid))
  # Independent manual aggregation on the shared mask, not separate map means.
  values <- vapply(out$cells[ids], function(x)
    mean(vapply(split(x$entropy[joint_mask], x$distance_index[joint_mask]), mean, numeric(1))), numeric(1))
  transform <- out$comparisons[out$comparisons$family == "transformed" & out$comparisons$metric == "pairwise", ]
  expect_equal(transform$observed_minus_control_mean, unname(values[1] - mean(values[-1])))
  expect_equal(transform$common_valid_cell_coverage, mean(joint_mask))
  expect_equal(out$features$observed_minus_transform_mean[1],
    out$features$first_order[1] - mean(out$features$first_order[out$features$family == "transformed"]))
  expect_true(all(is.na(out$features$observed_minus_transform_mean[-1])))

  symmetric <- compute_saliency_features(img, q*0+1, "weights", pairwise = FALSE,
                                        filter_length = 5L, gabor_bins = 4L)
  expect_equal(symmetric$comparisons$n_controls, c(0, 0))
  expect_true(all(is.na(symmetric$comparisons$observed_minus_control_mean)))
})
