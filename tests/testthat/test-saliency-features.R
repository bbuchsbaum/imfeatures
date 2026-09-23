saliency_test_image <- function() outer(seq_len(20), seq_len(21), function(x, y) sin(x*.7) + cos(y*.3))
saliency_test_args <- function() list(filter_length = 5L, gabor_bins = 4L,
  maxdiag = 18L, circ_bins = 4L, ranges = list(c(1L, 6L), c(7L, 12L), c(13L, 18L)))

test_that("uniform maps preserve native legacy results and agree across new backends", {
  img <- saliency_test_image()
  for (cpp in c(TRUE, FALSE)) {
    legacy <- do.call(compute_edge_entropy, c(list(image = img, use_cpp = cpp), saliency_test_args()))
    for (amplitude in c(1, 17)) {
      out <- do.call(compute_saliency_features, c(list(image = img,
        saliency = matrix(amplitude, nrow(img), ncol(img)), representation = "weights",
        use_cpp = cpp, transform_controls = FALSE), saliency_test_args()))
      expect_identical(out$features$first_order[1], legacy$entropy)
      if (cpp) expect_identical(out$ranges$entropy[out$ranges$control == "observed"],
                               unname(unlist(legacy[1, 3:5])))
      expect_identical(out$ranges$entropy[out$ranges$control == "observed"],
                       out$ranges$entropy[out$ranges$control == "unweighted"])
      expect_identical(out$metadata$geometry, "euclidean")
      expect_identical(out$selection$observed, out$selection$unweighted)
      expect_equal(out$cells$observed$effective_pairs_ratio[out$cells$observed$mass > 0],
                   rep(1, sum(out$cells$observed$mass > 0)), tolerance = 1e-12)
    }
  }
})

test_that("weighted results, support and masks are scale invariant", {
  img <- saliency_test_image()
  q <- outer(seq_len(nrow(img)), seq_len(ncol(img)), function(x, y) exp(-((x-7)^2+(y-9)^2)/25))
  run <- function(s, type = "weights", cpp = TRUE) do.call(compute_saliency_features,
    c(list(image = img, saliency = s, representation = type, use_cpp = cpp,
      transform_controls = FALSE, validity = "support", min_cell_mass_fraction = 0,
      min_effective_pairs = 0), saliency_test_args()))
  for (cpp in c(TRUE, FALSE)) {
    a <- run(q, cpp = cpp); b <- run(q * 1e100, cpp = cpp)
    expect_equal(a$features, b$features, tolerance = 1e-12)
    expect_equal(a$ranges, b$ranges, tolerance = 1e-12)
    expect_identical(a$cells$observed$valid, b$cells$observed$valid)
    expect_equal(a$cells, b$cells, tolerance = 1e-12)
    expect_equal(a$features, run(log(q) - 100, "log_density", cpp)$features, tolerance = 1e-12)
    # Controls share the image-selected edges even when their maps differ.
    expect_identical(a$selection$observed, a$selection$centre_bias)
  }
})

test_that("unavailable edge quantities remain distinct from valid zero entropy", {
  img <- saliency_test_image()
  run <- function(image, q, cpp = TRUE) do.call(compute_saliency_features,
    c(list(image = image, saliency = q, representation = "weights", use_cpp = cpp,
           transform_controls = FALSE), saliency_test_args()))
  for (cpp in c(TRUE, FALSE)) {
    out <- run(img, img * 0, cpp)
    expect_identical(out$features$first_order_status[1], "zero_map_mass")
    expect_true(is.na(out$features$first_order[1]))
    expect_true(all(is.na(out$ranges$entropy[out$ranges$control == "observed"])))
    expect_false(any(out$cells$observed$valid))
    blank <- run(img * 0, img * 0 + 1, cpp)
    expect_identical(blank$features$first_order_status[1], "no_edges")
    expect_equal(blank$features$effective_fraction[1], 1)
    edge_zero <- img * 0; edge_zero[1, 1] <- 1
    no_overlap <- run(img, edge_zero, cpp)
    expect_identical(no_overlap$features$first_order_status[1], "zero_edge_saliency")
    expect_equal(no_overlap$features$H_location[1], 0)
    legacy_zero <- suppressWarnings(do.call(compute_edge_entropy,
      c(list(image = img * 0, use_cpp = cpp), saliency_test_args())))
    expect_true(is.na(legacy_zero$entropy))
  }
})

test_that("public controls, multiscale output and validation are usable", {
  img <- saliency_test_image()
  assignment <- saliency_donor_assignments(c("a", "b"), rep("same", 2), replicates = 1)
  out <- do.call(compute_saliency_features, c(list(image = img, saliency = abs(img),
    representation = "weights", pairwise = FALSE, image_id = "a", scales = c(.25, .5),
    donor_maps = list(b = abs(img[nrow(img):1, ])), donor_assignments = assignment), saliency_test_args()))
  expect_true(all(c("observed", "unweighted", "centre_bias", "donor_1") %in% out$features$control))
  expect_equal(out$features$observed_minus_control,
               out$features$first_order[1] - out$features$first_order)
  expect_equal(nrow(out$multiscale$observed$curve), 8)
  expect_equal(nrow(out$ranges), 0)
  expect_length(out$cells, 0)
  expect_error(compute_saliency_features(img, abs(img), "weights", validity = "support"), "number")
  expect_error(compute_saliency_features(img, matrix(1, 2, 2), "weights"), "dimensions")
})

test_that("file inputs resize an asymmetric map on the image grid", {
  path <- tempfile(fileext = ".png")
  on.exit(unlink(path))
  x <- outer(seq_len(10), seq_len(20), function(i, j) (i + j)/30)
  rgb <- imager::imappend(list(imager::as.cimg(x), imager::as.cimg(x^2),
                              imager::as.cimg(sqrt(x))), "c")
  imager::save.image(rgb, path)
  sal <- matrix(0, 10, 20); sal[2:4, 14:19] <- 1
  run <- function(cpp) compute_saliency_features(path, sal, "weights", max_pixels = 50L,
    pairwise = FALSE, filter_length = 3L, gabor_bins = 4L,
    transform_controls = FALSE, use_cpp = cpp)
  native <- run(TRUE); reference <- run(FALSE)
  expect_identical(as.integer(native$metadata$original_dimensions), c(10L, 20L))
  expect_identical(as.integer(native$metadata$analysis_dimensions), c(5L, 10L))
  expect_identical(native$metadata$map_interpolation, "area")
  expect_equal(native$metadata$controls$observed$location_before_resize,
               saliency_location_entropy(sal, "weights"))
  expect_equal(native$metadata$controls$observed$location_after_resize$H_location,
               native$features$H_location[1])
  expect_equal(native$features$H_location, reference$features$H_location, tolerance = 1e-12)
  expect_true(is.finite(native$features$H_location[1]))
  legacy <- compute_edge_entropy(path, max_pixels = 50L, filter_length = 3L, gabor_bins = 4L)
  loaded <- filtered_image(path, max_pixels = 50L)
  matrix_result <- compute_edge_entropy(matrix(loaded$image_raw, 5L, 10L),
                                       filter_length = 3L, gabor_bins = 4L)
  expect_identical(legacy$entropy, matrix_result$entropy)
  uniform <- compute_saliency_features(path, sal * 0 + 1, "weights", max_pixels = 50L,
    pairwise = FALSE, filter_length = 3L, gabor_bins = 4L, transform_controls = FALSE)
  expect_identical(uniform$features$first_order[1], legacy$entropy)
  expect_error(compute_saliency_features(path, t(sal), "weights"), "dimensions")
})
