library(testthat)
library(imfeatures)

context("im_feature_sim")
# with_mocked_bindings defined in helper-mocks.R


test_that("subsampling logic works and output matrices are symmetric", {
  recorded_dim <- NULL
  mock_cosine <- function(mat) {
    recorded_dim <<- dim(mat)
    n <- nrow(mat)
    diag(n)
  }

  res <- with_mocked_bindings(
    im_features = function(impath, layers, ...) {
      lapply(layers, function(l) seq_len(20))
    },
    `progress_bar$new` = function(total) { list(tick = function(){}) },
    `memoise::memoise` = function(f, ...) f,
    coop::tcosine = mock_cosine,
    furrr::future_map = function(x, f) lapply(x, f),
    sample = function(x, size, ...) seq_len(size),
    im_feature_sim(c("img1", "img2", "img3"), layers = 1, lowmem = FALSE, subsamp_prop = 0.5, model = list())
  )

  expect_equal(recorded_dim[2], 10)
  expect_true(all(sapply(res, isSymmetric)))
})

test_that("spatial_pooling argument is passed to im_features", {
  seen <- NULL
  with_mocked_bindings(
    im_features = function(impath, layers, spatial_pooling = "none", ...) {
      seen <<- spatial_pooling
      lapply(layers, function(l) l)
    },
    `progress_bar$new` = function(total) { list(tick = function(){}) },
    `memoise::memoise` = function(f, ...) f,
    coop::tcosine = function(mat) diag(nrow(mat)),
    furrr::future_map = function(x, f) lapply(x, f),
    sample = function(x, size, ...) seq_len(size),
    im_feature_sim(c("img1", "img2"), layers = 1, lowmem = FALSE,
                   spatial_pooling = "avg", model = list())
  )
  expect_equal(seen, "avg")
})


