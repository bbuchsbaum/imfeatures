library(testthat)

context("hierarchy extraction spec and mocked VGG path")

test_that("VGG-16 and VGG-19 registries use block-end taps", {
  t16 <- vgg_block_end_taps("vgg16")
  t19 <- vgg_block_end_taps("vgg19")
  expect_equal(unname(t16[c("b1", "b3", "b5")]), c("block1_conv2", "block3_conv3", "block5_conv3"))
  expect_equal(unname(t19[c("b3", "b4", "b5")]), c("block3_conv4", "block4_conv4", "block5_conv4"))
  expect_identical(unname(t16[["logits"]]), "classifier_logits")
})

test_that("vgg16_block_end_spec uses depth-adaptive DCT poolers", {
  spec <- vgg16_block_end_spec()
  expect_s3_class(spec, "vgg_hierarchy_spec")
  expect_equal(spec$pooling$b1$type, "dct")
  expect_equal(spec$pooling$b1$ny, 6L)
  expect_equal(spec$pooling$b5$nx, 2L)
  expect_equal(spec$pooling$fc1$type, "identity")
  expect_equal(spec$pooling$logits$type, "identity")
})

test_that("dense logits are an affine map, not a softmax", {
  set.seed(1)
  fc2 <- matrix(rnorm(4 * 3), 4, 3)
  W <- matrix(rnorm(3 * 5), 3, 5)
  b <- rnorm(5)
  logits <- imfeatures:::.dense_logits(fc2, W, b)
  expect_equal(logits, sweep(fc2 %*% W, 2, b, `+`), tolerance = 1e-12)
  expect_true(any(logits < 0))
  probs <- exp(logits) / rowSums(exp(logits))
  expect_false(isTRUE(all.equal(logits, probs)))
})

test_that("extract_feature_hierarchy returns named blocks under mocks", {
  state <- new.env(parent = emptyenv())
  state$build_calls <- 0L
  local_mocked_bindings(
    .vgg_multi_output_model = function(model, layers) {
      state$build_calls <- state$build_calls + 1L
      structure(list(layers = layers), class = "mock_multi_model")
    },
    .vgg_forward_batch = function(multi, paths, target_size) {
      n <- length(paths)
      lapply(multi$layers, function(layer) {
        if (layer %in% c("fc1", "fc2")) {
          matrix(seq_len(n * 4), n, 4)
        } else {
          array(seq_len(n * 4 * 4 * 2), dim = c(n, 4, 4, 2))
        }
      })
    },
    .package = "imfeatures"
  )

  img_dir <- file.path(tempdir(), "hier_extract")
  dir.create(img_dir, recursive = TRUE, showWarnings = FALSE)
  paths <- file.path(img_dir, paste0("img", 1:3, ".png"))
  for (p in paths) file.create(p)

  spec <- vgg_hierarchy_spec(
    architecture = "vgg16",
    taps = c(b1 = "block1_conv2", fc1 = "fc1"),
    pooling = list(b1 = dct_pool(2, 2), fc1 = identity_pool())
  )
  raw <- extract_feature_hierarchy(paths, spec = spec, model = list(dummy = TRUE), batch_size = 2L)
  expect_s3_class(raw, "feature_hierarchy")
  expect_equal(raw$stage_order, c("b1", "fc1"))
  expect_equal(nrow(raw$blocks$b1), 3)
  expect_equal(ncol(raw$blocks$b1), 2 * 2 * 2)
  expect_equal(ncol(raw$blocks$fc1), 4)
  expect_true(all(grepl("^b1::channel_", colnames(raw$blocks$b1))))
  expect_identical(state$build_calls, 1L)
})

test_that("extract_feature_hierarchy rejects a missing spec", {
  expect_error(
    extract_feature_hierarchy("nope.png", spec = list()),
    "vgg_hierarchy_spec"
  )
})

test_that("extracted blocks are named to match their pooler and tap", {
  local_mocked_bindings(
    .vgg_multi_output_model = function(model, layers) {
      structure(list(layers = layers), class = "mock_multi_model")
    },
    .vgg_forward_batch = function(multi, paths, target_size) {
      n <- length(paths)
      lapply(multi$layers, function(layer) {
        if (layer %in% c("fc1", "fc2")) {
          matrix(seq_len(n * 4), n, 4)
        } else {
          array(seq_len(n * 2 * 3 * 2), dim = c(n, 2, 3, 2))
        }
      })
    },
    .package = "imfeatures"
  )

  img_dir <- file.path(tempdir(), "hier_extract_names")
  dir.create(img_dir, recursive = TRUE, showWarnings = FALSE)
  paths <- file.path(img_dir, paste0("img", 1:3, ".png"))
  for (p in paths) file.create(p)

  spec <- vgg_hierarchy_spec(
    architecture = "vgg16",
    taps = c(flat = "block5_conv3", grid = "block4_conv3", fc1 = "fc1"),
    pooling = list(
      flat = identity_pool(),
      grid = adaptive_grid_pool(1, 3),
      fc1 = global_average_pool()
    )
  )
  raw <- extract_feature_hierarchy(paths, spec = spec, model = list(dummy = TRUE), batch_size = 2L)

  # A flattened map names every position: height fastest, then width, channel.
  expect_equal(ncol(raw$blocks$flat), 2 * 3 * 2)
  expect_equal(
    colnames(raw$blocks$flat)[c(1, 2, 3, 7)],
    c(
      "flat::channel_001::y1_x1", "flat::channel_001::y2_x1",
      "flat::channel_001::y1_x2", "flat::channel_002::y1_x1"
    )
  )
  expect_equal(
    colnames(raw$blocks$grid)[1:4],
    c(
      "grid::channel_001::grid_y1_x1", "grid::channel_001::grid_y1_x2",
      "grid::channel_001::grid_y1_x3", "grid::channel_002::grid_y1_x1"
    )
  )
  # A dense tap has no channels or positions, whatever its pooler.
  expect_equal(colnames(raw$blocks$fc1), sprintf("fc1::f_%03d", 1:4))
})

test_that("a single pooler is recycled and the spec prints its pooling", {
  spec <- vgg_hierarchy_spec("vgg16", pooling = adaptive_grid_pool(2, 2))
  expect_true(all(vapply(spec$pooling, function(p) p$type, character(1)) == "grid"))
  expect_named(spec$pooling, names(spec$taps))

  default <- vgg_hierarchy_spec("vgg16")
  expect_equal(default$pooling$b1$type, "gap")
  expect_equal(default$pooling$fc1$type, "identity")

  expect_error(
    vgg_hierarchy_spec("vgg16", pooling = list(b1 = dct_pool(2, 2))),
    "Pooling is missing stages"
  )
  out <- capture.output(print(vgg16_block_end_spec()))
  expect_true(any(grepl("b1 +DCT pool 6 x 6", out)))
  expect_true(any(grepl("fc1 +Identity pool", out)))
})
