test_that("donor assignments are valid permutations and preserve caller RNG", {
  ids <- c("a", "b", "c", "d", "alone")
  strata <- c(rep("square-model1", 4), "rectangle-model1")
  stimuli <- c("a", "a", "c", "d", "alone")
  set.seed(33)
  before <- .Random.seed
  out <- saliency_donor_assignments(ids, strata, stimuli, replicates = 4L, seed = 20L)
  expect_identical(.Random.seed, before)
  expect_identical(out, saliency_donor_assignments(ids, strata, stimuli, 4L, 20L))
  ok <- out[out$status == "ok", ]
  expect_true(all(ok$target_id != ok$donor_id))
  expect_true(all(ok$target_stimulus_id != ok$donor_stimulus_id))
  for (r in 1:4) expect_setequal(ok$donor_id[ok$replicate == r], ids[1:4])
  expect_true(all(out$status[out$target_id == "alone"] == "no_valid_permutation"))
  expect_true(all(is.na(out$donor_id[out$target_id == "alone"])))
  impossible <- saliency_donor_assignments(c("a", "b", "c"), rep("x", 3), c("x", "x", "y"))
  expect_true(all(impossible$status == "no_valid_permutation"))
})

test_that("entropy-matched centre controls disclose attained and impossible fits", {
  q <- outer(1:30, 1:40, function(x, y) exp(-((x-10)^2+(y-13)^2)/30))
  q <- q / sum(q)
  fit <- .saliency_match_centre(q, "ok")
  expect_identical(fit$status, "ok")
  expect_equal(.saliency_location(fit$q)$effective_fraction,
               .saliency_location(q)$effective_fraction, tolerance = 1e-9)
  expect_equal(fit$q, fit$q[30:1, ], tolerance = 1e-12)
  expect_equal(fit$q, fit$q[, 40:1], tolerance = 1e-12)
  point <- q * 0; point[1, 1] <- 1
  impossible <- .saliency_match_centre(point, "ok")
  expect_identical(impossible$status, "unattainable_effective_fraction")
  expect_equal(impossible$metadata$minimum_effective_fraction, 4 / length(q))
  uniform <- .saliency_match_centre(q*0+1/length(q), "ok")
  expect_equal(uniform$q, q*0+1/length(q))
  expect_identical(uniform$metadata$sigma_image_fraction, c(Inf, Inf))
  expect_identical(.saliency_match_centre(q*0, "zero_map_mass")$status, "zero_map_mass")
})

test_that("donor aspect ratios permit declared small rounding differences", {
  q <- matrix(1/48, 8, 6)
  assignment <- saliency_donor_assignments(c("a", "b"), rep("same", 2), replicates = 1)
  donor <- matrix(1, 1023, 768)
  out <- .saliency_controls(q, "ok", .25, FALSE, list(b = donor), assignment,
                            "a", "weights", "reject", c(1024, 768))
  expect_identical(out$donor_1$status, "ok")
  expect_equal(out$donor_1$metadata$aspect_ratio_relative_error, 1/1024)
  expect_identical(out$donor_1$metadata$map_interpolation, "area")
  expect_error(.saliency_controls(q, "ok", .25, FALSE, list(b = donor), assignment,
    "a", "weights", "reject", c(1024, 768), donor_aspect_tolerance = 0), "aspect ratio")
})

test_that("grid transforms preserve mass and flag symmetric degeneracy", {
  q <- matrix(1:12, 3, 4); q <- q / sum(q)
  transformed <- .saliency_transforms(q)
  expect_length(transformed, 3)
  for (x in transformed) {
    expect_identical(sort(as.vector(x)), sort(as.vector(q)))
    expect_equal(.saliency_location(x), .saliency_location(q), tolerance = 1e-12)
  }
  q <- matrix(1/9, 3, 3)
  controls <- .saliency_controls(q, "ok", .25, TRUE, list(), NULL,
                                "image", "weights", "reject", c(3L, 3L))
  expect_length(.saliency_transforms(q), 5)
  expect_true(controls$rotate90$identity_equivalent)
  expect_identical(controls$rotate90$duplicate_of, "observed")
  expect_identical(controls$donor_unavailable$status, "no_donor_pool")
})

test_that("donor controls disclose resize effects and forbid self donors", {
  q <- matrix(1/16, 4, 4)
  assignments <- saliency_donor_assignments(c("a", "b"), rep("square", 2), replicates = 1)
  donor <- matrix(c(1, 0, 0, 0), 2)
  out <- .saliency_controls(q, "ok", .25, FALSE, list(b = donor), assignments,
                            "a", "weights", "reject", c(4L, 4L))
  expect_equal(sum(out$donor_1$q), 1)
  expect_true(out$donor_1$metadata$resized)
  expect_identical(out$donor_1$metadata$source_dimensions, c(2L, 2L))
  expect_true(is.data.frame(out$donor_1$metadata$location_before_resize))
  assignments$donor_id[assignments$target_id == "a"] <- "a"
  expect_error(.saliency_controls(q, "ok", .25, FALSE, list(a = donor), assignments,
    "a", "weights", "reject", c(4L, 4L)), "Self")
})
