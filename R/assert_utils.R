#' Internal assertion helpers using checkmate
#'
#' These small wrappers standardize argument checks across
#' the package and yield clear error messages.
#'
#' @keywords internal
assert_scalar <- function(x, type = c("character", "numeric", "integer", "logical"),
                          na.ok = FALSE, .var.name = deparse(substitute(x))) {
  type <- match.arg(type)
  switch(type,
    character = checkmate::assert_character(x,
      len = 1, any.missing = na.ok,
      .var.name = .var.name
    ),
    numeric = checkmate::assert_number(x,
      na.ok = na.ok,
      .var.name = .var.name
    ),
    integer = checkmate::assert_integer(x,
      len = 1, any.missing = na.ok,
      .var.name = .var.name
    ),
    logical = checkmate::assert_logical(x,
      len = 1, any.missing = na.ok,
      .var.name = .var.name
    )
  )
  invisible(TRUE)
}

#' @keywords internal
assert_image <- function(image, .var.name = deparse(substitute(image))) {
  if (is.character(image)) {
    checkmate::assert_character(image,
      min.len = 1, any.missing = FALSE,
      .var.name = .var.name
    )
    missing <- image[!file.exists(image)]
    if (length(missing)) {
      stop(sprintf(
        "%s file(s) not found: %s", .var.name,
        paste(missing, collapse = ", ")
      ))
    }
  } else if (inherits(image, "cimg")) {
    # no further checks
  } else if (is.matrix(image) || is.array(image)) {
    checkmate::assert_numeric(image,
      any.missing = TRUE,
      .var.name = .var.name
    )
  } else {
    stop(sprintf(
      "%s must be a file path, cimg object, or numeric matrix/array",
      .var.name
    ))
  }
  invisible(TRUE)
}
