test_that("CLIP extraction uses validation preprocessing (regression #51)", {
  path <- tempfile(fileext = ".png")
  file.create(path)
  on.exit(unlink(path), add = TRUE)

  state <- new.env(parent = emptyenv())
  state$evaluated <- FALSE
  state$training_calls <- 0L
  state$validation_calls <- 0L
  image <- list(convert = function(mode) {
    expect_identical(mode, "RGB")
    "rgb-image"
  })
  model <- list(eval = function() state$evaluated <- TRUE)
  modules <- list(
    open_clip = list(create_model_and_transforms = function(...) {
      list(
        model,
        function(image) {
          state$training_calls <- state$training_calls + 1L
          stop("TRAIN_TRANSFORM_SELECTED")
        },
        function(image) {
          state$validation_calls <- state$validation_calls + 1L
          expect_true(state$evaluated)
          expect_identical(image, "rgb-image")
          stop("VALIDATION_TRANSFORM_SELECTED")
        }
      )
    }),
    PIL.Image = list(open = function(impath) {
      expect_identical(impath, path)
      image
    }),
    torch = list()
  )
  # `reticulate::py` is a read-only active binding; restore it after the test.
  ns <- asNamespace("reticulate")
  original_py <- activeBindingFunction("py", ns)
  unlockBinding("py", ns)
  makeActiveBinding("py", function() list(), ns)
  lockBinding("py", ns)
  on.exit({
    unlockBinding("py", ns)
    makeActiveBinding("py", original_py, ns)
    lockBinding("py", ns)
  }, add = TRUE)

  local_mocked_bindings(
    py_run_string = function(...) invisible(NULL),
    import = function(module, ...) modules[[module]],
    .package = "reticulate"
  )

  expect_error(
    clip_features(path, num_transformer_blocks = 12L, device = "cpu"),
    "VALIDATION_TRANSFORM_SELECTED"
  )
  expect_identical(state$training_calls, 0L)
  expect_identical(state$validation_calls, 1L)
})
