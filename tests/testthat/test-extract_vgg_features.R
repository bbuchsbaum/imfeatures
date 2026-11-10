library(testthat)

context("extract_vgg_features")

# extract_vgg_features() runs one multi-output model over batches of images
# rather than one model per layer per image, so the tests substitute the two
# internal seams (model construction and the batched forward pass) instead of
# mocking im_features().

local_vgg_mocks <- function(state = new.env(parent = emptyenv()),
                            env = parent.frame()) {
  state$build_calls <- 0L
  state$predict_calls <- 0L
  state$batch_sizes <- integer()

  local_mocked_bindings(
    .vgg_multi_output_model = function(model, layers) {
      state$build_calls <- state$build_calls + 1L
      structure(list(layers = layers), class = "mock_multi_model")
    },
    .vgg_forward_batch = function(multi, paths, target_size) {
      ids <- as.numeric(sub("^img([0-9]+)\\.png$", "\\1", basename(paths)))
      state$predict_calls <- state$predict_calls + 1L
      state$batch_sizes <- c(state$batch_sizes, length(paths))

      lapply(seq_along(multi$layers), function(layer_index) {
        cbind(
          ids + layer_index * 100,
          ids + layer_index * 100 + 0.5
        )
      })
    },
    .package = "imfeatures",
    .env = env
  )
  invisible(state)
}

make_dummy_images <- function(dir, n = 2) {
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  paths <- file.path(dir, paste0("img", seq_len(n), ".png"))
  for (p in paths) file.create(p)
  paths
}

test_that("directory input expands to images and returns expected dims", {
  local_vgg_mocks()
  img_dir <- file.path(tempdir(), "imgs")
  make_dummy_images(img_dir, 2)
  res <- extract_vgg_features(img_dir, tier = "low", model = list(dummy = TRUE))
  expect_s3_class(res, "vgg_feature_set")
  expect_equal(length(res$image_paths), 2)
  expect_equal(nrow(res$features), 2)
  expect_equal(ncol(res$features), length(res$layer_names) * 2)
})

test_that("error for nonexistent image paths", {
  expect_error(
    extract_vgg_features(c("no_such_file1.png", "no_such_file2.png"),
                        model = list(dummy = TRUE)),
    "impaths file\\(s\\) not found: no_such_file1.png, no_such_file2.png"
  )
})

test_that("returns correct class and dims for explicit image paths", {
  local_vgg_mocks()
  img_dir <- file.path(tempdir(), "imgs2")
  paths <- make_dummy_images(img_dir, 3)
  res <- extract_vgg_features(paths[1:2], tier = "low", model = list(dummy = TRUE))
  expect_s3_class(res, "vgg_feature_set")
  expect_equal(nrow(res$features), 2)
  expect_equal(ncol(res$features), length(res$layer_names) * 2)
})

test_that("features matrix is numeric, not character (regression #45)", {
  local_vgg_mocks()
  img_dir <- file.path(tempdir(), "imgs_numeric")
  paths <- make_dummy_images(img_dir, 2)
  res <- extract_vgg_features(paths, tier = "low", model = list(dummy = TRUE))
  expect_true(is.numeric(res$features))
  expect_equal(typeof(res$features), "double")
  expect_false(any(is.character(res$features)))
  expect_equal(nrow(res$features), length(paths))
})

test_that("chunking preserves image order and limits forward passes", {
  state <- local_vgg_mocks()
  img_dir <- file.path(tempdir(), "imgs_batch")
  paths <- make_dummy_images(img_dir, 5)
  out <- extract_vgg_features(
    paths,
    tier = "low",
    model = list(dummy = TRUE),
    batch_size = 2L
  )

  expected <- do.call(cbind, lapply(seq_along(out$layer_names), function(layer_index) {
    cbind(
      seq_along(paths) + layer_index * 100,
      seq_along(paths) + layer_index * 100 + 0.5
    )
  }))

  expect_equal(out$features, expected)
  expect_identical(state$build_calls, 1L)
  expect_identical(state$predict_calls, 3L)
  expect_identical(state$batch_sizes, c(2L, 2L, 1L))
  expect_identical(out$batch_size, 2L)
})

test_that("batch size does not change feature layout (regression #46)", {
  img_dir <- file.path(tempdir(), "imgs_batch_equivalence")
  paths <- make_dummy_images(img_dir, 5)

  local_vgg_mocks()
  one <- extract_vgg_features(
    paths,
    tier = "low",
    model = list(dummy = TRUE),
    batch_size = 1L
  )

  big <- extract_vgg_features(
    paths,
    tier = "low",
    model = list(dummy = TRUE),
    batch_size = 8L
  )

  expect_identical(one$features, big$features)
  expect_equal(nrow(one$features), 5)
  expect_identical(big$batch_size, 8L)
})

test_that("batch_size is validated", {
  img_dir <- file.path(tempdir(), "imgs_bs")
  paths <- make_dummy_images(img_dir, 2)
  expect_error(
    extract_vgg_features(paths, tier = "low", model = list(dummy = TRUE), batch_size = 0)
  )
})

test_that("the default batch size bounds low-tier activation memory", {
  expect_identical(formals(extract_vgg_features)$batch_size, 8L)
})

test_that("malformed multi-output predictions fail explicitly", {
  expect_error(
    imfeatures:::.validate_vgg_batch_outputs(
      list(matrix(1, nrow = 2, ncol = 1)),
      expected_layers = 2L,
      batch_n = 2L
    ),
    "Expected 2 VGG layer output"
  )

  expect_error(
    imfeatures:::.validate_vgg_batch_outputs(
      list(
        matrix(1, nrow = 2, ncol = 1),
        matrix(1, nrow = 1, ncol = 1)
      ),
      expected_layers = 2L,
      batch_n = 2L
    ),
    "batch dimension"
  )
})

test_that("pooling accepts any resize_HxW and rejects malformed values", {
  img_dir <- file.path(tempdir(), "imgs_pooling_arg")
  paths <- make_dummy_images(img_dir, 2)
  local_vgg_mocks()

  # The mocked layers are dense, so pooling leaves them alone; this checks
  # that the argument is accepted and recorded.
  for (pooling in c("none", "avg", "max", "resize_3x3", "resize_2x4")) {
    out <- extract_vgg_features(
      paths,
      tier = "high", model = list(dummy = TRUE), pooling = pooling
    )
    expect_identical(out$pooling, pooling)
  }

  for (pooling in c("mean", "resize_0x3", "resize_3", "resize_axb")) {
    expect_error(
      extract_vgg_features(
        paths,
        tier = "high", model = list(dummy = TRUE), pooling = pooling
      ),
      "'pooling' must be"
    )
  }
})

test_that("pooling reduces 4-D maps per image and layer", {
  img_dir <- file.path(tempdir(), "imgs_pooling_values")
  paths <- make_dummy_images(img_dir, 3)

  maps <- function(n, layer_index) {
    a <- array(0, dim = c(n, 2, 2, 2))
    for (i in seq_len(n)) {
      a[i, , , 1] <- i * c(1, 2, 3, 4)
      a[i, , , 2] <- layer_index * 10 + i * c(4, 3, 2, 1)
    }
    a
  }
  local_mocked_bindings(
    .vgg_multi_output_model = function(model, layers) {
      structure(list(layers = layers), class = "mock_multi_model")
    },
    .vgg_forward_batch = function(multi, paths, target_size) {
      lapply(seq_along(multi$layers), function(j) maps(length(paths), j))
    },
    .package = "imfeatures"
  )

  avg <- extract_vgg_features(paths, tier = "high", model = list(dummy = TRUE), pooling = "avg")
  mx <- extract_vgg_features(paths, tier = "high", model = list(dummy = TRUE), pooling = "max")
  raw <- extract_vgg_features(paths, tier = "high", model = list(dummy = TRUE), pooling = "none")

  # Three layers of two channels each: channel-wise summaries, layer by layer.
  expect_equal(dim(avg$features), c(3, 6))
  expect_equal(avg$features[2, ], c(5, 15, 5, 25, 5, 35))
  expect_equal(mx$features[3, ], c(12, 22, 12, 32, 12, 42))

  # "none" flattens each layer with height fastest, then width, then channel.
  expect_equal(dim(raw$features), c(3, 3 * 2 * 2 * 2))
  expect_equal(raw$features[1, 1:8], c(1, 2, 3, 4, 14, 13, 12, 11))
})
