#' Multiscale entropy for images
#'
#' Compute the entropy of the Hue, Saturation and Value components of an image
#' across multiple blur scales.
#'
#' @importFrom imager RGBtoHSV imsplit isoblur add.colour channels
#' @importFrom entropy discretize freqs entropy
#' @param im image of type `cimg` from `imager` package. The function expects an
#'   image with three colour channels. If a single-channel image is supplied it
#'   will be converted with \code{add.colour()}, and an error is thrown if the
#'   result does not contain exactly three channels.
#' @param sf vector smoothing factors indicating the scales for entropy computation
#' @param bins number of bins for computing information
#' @return A named numeric vector with the mean multiscale entropy for the
#'   \code{H}, \code{S} and \code{V} channels.
#' @examples
#' \dontrun{
#' # Load an example image
#' library(imager)
#' img <- load.example("parrots")
#'
#' # Compute multiscale entropy with default parameters
#' mse_values <- image_mse(img)
#' print(mse_values) # H, S, V entropy values
#'
#' # Use custom smoothing factors for different scales
#' mse_custom <- image_mse(
#'   im = img,
#'   sf = c(200, 100, 50, 25, 0), # More smoothing levels
#'   bins = 32L # More bins for finer discretization
#' )
#'
#' # Process a grayscale image (will be converted to color)
#' gray_img <- grayscale(img)
#' mse_gray <- image_mse(gray_img)
#' # Note: H and S channels will be 0 for grayscale images
#'
#' # Batch process multiple images
#' img_files <- list.files("path/to/images", pattern = "\\.jpg$", full.names = TRUE)
#' mse_results <- lapply(img_files, function(f) {
#'   img <- load.image(f)
#'   image_mse(img, sf = c(50, 25, 0))
#' })
#'
#' # Convert to data frame for analysis
#' mse_df <- do.call(rbind, mse_results)
#' colMeans(mse_df) # Average entropy across images
#' }
#' @export
image_mse <- function(im, sf = c(100, 50, 8, 4, 0), bins = 16L) {
  assert_image(im)
  checkmate::assert_integerish(sf, min.len = 1)
  assert_scalar(bins, "integer")

  # Track if input was grayscale without NA values
  was_grayscale <- length(channels(im)) == 1
  has_na <- anyNA(as.vector(im))

  if (was_grayscale) {
    im <- add.colour(im)
  }
  if (length(channels(im)) != 3) {
    stop("image_mse expects an image with 3 colour channels after add.colour()")
  }

  hsvim <- RGBtoHSV(im)
  hsvim2 <- imsplit(hsvim, "c")

  ret <- lapply(sf, function(fac) {
    i1 <- imager::isoblur(hsvim2[[1]], fac)
    i2 <- imager::isoblur(hsvim2[[2]], fac)
    i3 <- imager::isoblur(hsvim2[[3]], fac)
    f1 <- try(freqs(entropy::discretize(as.vector(i1), bins)), silent = TRUE)
    f2 <- try(freqs(entropy::discretize(as.vector(i2), bins)), silent = TRUE)
    f3 <- try(freqs(entropy::discretize(as.vector(i3), bins)), silent = TRUE)

    e1 <- if (!inherits(f1, "try-error")) {
      entropy(f1)
    } else {
      NA
    }

    e2 <- if (!inherits(f2, "try-error")) {
      entropy(f2)
    } else {
      NA
    }

    e3 <- if (!inherits(f3, "try-error")) {
      entropy(f3)
    } else {
      NA
    }

    data.frame(fac = fac, e1 = e1, e2 = e2, e3 = e3)
  })

  ret <- do.call(rbind, ret)
  out <- colMeans(as.matrix(ret[, 2:4]), na.rm = TRUE)

  # For grayscale images without NA, replace NaN (from undefined H/S) with 0
  # For images with NA values, preserve NA in the output
  if (was_grayscale && !has_na) {
    out[is.nan(out)] <- 0
  }

  names(out) <- c("H", "S", "V")
  out
}
