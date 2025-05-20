#' Multiscale entropy for images
#'
#'
#' @import imager
#' @import entropy
#' @param im image of type `cimg` from `imager` package
#' @param sf vector smoothing factors indicating the scales for entropy computation
#' @param bins number of bins for computing information
#' @export
image_mse <- function(im, sf=c(100, 50, 8, 4, 0), bins=16) {
  if (length(channels(im)) ==1) {
    im <- add.colour(im)
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
    } else NA

    e2 <- if (!inherits(f2, "try-error")) {
      entropy(f2)
    } else NA

    e3 <- if (!inherits(f3, "try-error")) {
      entropy(f3)
    } else NA

    data.frame(fac=fac, e1=e1,e2=e2, e3=e3)
  })

  ret <- do.call(rbind, ret)
  colMeans(as.matrix(ret[, 2:4]), na.rm = TRUE)
}
