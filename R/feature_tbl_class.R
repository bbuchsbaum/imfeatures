#' Create a feature tibble
#'
#' Internal helper to mark a tibble of extracted features so it
#' prints nicely and works with dplyr verbs.
#'
#' @param x A tibble.
#' @return A tibble with class `imfeatures_feature_tbl`.
#' @keywords internal
new_feature_tbl <- function(x) {
  class(x) <- c("imfeatures_feature_tbl", class(x))
  x
}

#' @export
print.imfeatures_feature_tbl <- function(x, ...) {
  print(tibble::as_tibble(x), ...)
  invisible(x)
}
