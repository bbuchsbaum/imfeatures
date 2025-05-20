library(testthat)
# library(imfeatures) # Assuming devtools::test() or similar loads the package

context("im_feature_sim")

# The user's alias `with_mocked_bindings` pointed to withr::local_bindings, which is not exported.
# We will use testthat::with_mock instead.

test_that("subsampling logic works and output matrices are symmetric", {
  recorded_dim <- NULL
  # This mock_cosine will receive the matrix that coop::tcosine would process.
  # If features are rows and images are columns, dim(mat) = c(n_features, n_images)
  mock_cosine_func <- function(mat) { 
    recorded_dim <<- dim(mat)
    # coop::tcosine returns an n_images x n_images similarity matrix
    # For this test, we need to ensure the output is symmetric and has correct dims for downstream checks.
    # If mat is n_features x n_cols (n_images), tcosine output is n_cols x n_cols.
    # The mock should return a symmetric matrix of size ncol(mat) x ncol(mat).
    num_cols <- ncol(mat)
    if (is.null(num_cols)) { # if mat is a vector (single image after subsampling in some edge case)
        num_cols <- 1
        # This case should ideally not happen if tcosine expects a matrix with >1 col for similarity
        # For now, let's assume mat will have multiple columns (images)
        return(matrix(1, 1, 1)) 
    }
    sym_matrix <- matrix(0, num_cols, num_cols)
    diag(sym_matrix) <- 1
    # Make it symmetric for the test expect_true(all(sapply(res, isSymmetric)))
    # This simple diag matrix is already symmetric.
    return(sym_matrix) 
  }

  # Mock for imfeatures::im_features
  # It's called with (impath, layer_name, ...)
  # Should return list(layer_name = feature_vector_of_length_20)
  mock_im_features_func <- function(impath, layers, ...) {
    if (length(layers) != 1) stop("Mock for im_features expects a single layer name here.")
    setNames(list(seq_len(20)), as.character(layers))
  }
  
  # Mock for progress::progress_bar$new()
  # We mock the progress_bar object generator itself
  mock_progress_package_obj <- list(
    progress_bar = list(
      new = function(total, ...) list(tick = function(...){})
    )
  )

  # Mock for memoise::memoise
  mock_memoise_func <- function(f, ...) { f } # Pass through

  # Mock for furrr::future_map
  mock_furrr_future_map_func <- function(.x, .f, ...) { lapply(.x, .f, ...) } # Sequential map for testing

  # Mock for base::sample
  mock_sample_func <- function(x, size, ...) { seq_len(size) } # Deterministic sample

  # Use testthat::with_mock
  # The first argument is the expression to evaluate with mocks.
  # Subsequent arguments are name-value pairs for mocks.
  res <- testthat::with_mock(
    # Expression to evaluate:
    im_feature_sim(c("img1", "img2", "img3"), layers = "1", lowmem = FALSE, subsamp_prop = 0.5, model = list()),
    
    # Mocks:
    im_features = mock_im_features_func, 
    `memoise::memoise` = mock_memoise_func,
    `coop::tcosine` = mock_cosine_func,
    `furrr::future_map` = mock_furrr_future_map_func,
    sample = mock_sample_func,
    `progress::progress_bar` = mock_progress_package_obj$progress_bar
    # .env argument is specific to testthat::with_mock if needed, but usually defaults correctly.
  )
  
  expect_equal(recorded_dim[1], 10) 
  expect_equal(recorded_dim[2], 3)  
  
  expect_type(res, "list")
  expect_length(res, 1)
  expect_true(!is.null(names(res)) && names(res)[1] == "1")
  sim_matrix <- res[[1]]
  expect_true(is.matrix(sim_matrix))
  expect_equal(dim(sim_matrix), c(3, 3)) 
  expect_true(isSymmetric(sim_matrix))
})


