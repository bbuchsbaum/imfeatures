#' Convert a 4-D activation tensor to NHWC layout
#'
#' @param a Numeric array. 4-D tensors are permuted; other ranks are returned
#'   unchanged.
#' @param layout \code{"nhwc"} (Keras, default) or \code{"nchw"} (PyTorch).
#'
#' @return The same array in \code{N × H × W × C} order when \code{a} is 4-D.
#' @export
as_nhwc <- function(a, layout = c("nhwc", "nchw")) {
  layout <- match.arg(layout)
  d <- dim(a)
  if (is.null(d) || length(d) != 4L) {
    return(a)
  }
  if (identical(layout, "nchw")) {
    return(aperm(a, c(1L, 3L, 4L, 2L)))
  }
  a
}

#' Global average pooling over spatial axes
#' @export
global_average_pool <- function() {
  structure(list(type = "gap"), class = "imfeatures_pooler")
}

#' Global max pooling over spatial axes
#' @export
global_max_pool <- function() {
  structure(list(type = "gmp"), class = "imfeatures_pooler")
}

#' Adaptive rectangular grid pooling
#'
#' @param ny,nx Number of bins along height and width.
#' @export
adaptive_grid_pool <- function(ny, nx) {
  checkmate::assert_count(ny, positive = TRUE)
  checkmate::assert_count(nx, positive = TRUE)
  structure(
    list(type = "grid", ny = as.integer(ny), nx = as.integer(nx)),
    class = "imfeatures_pooler"
  )
}

#' Low-frequency 2-D DCT pooling
#'
#' @param ny,nx Number of retained DCT modes along height and width.
#' @param order \code{"zigzag"} (default) or \code{"raster"}.
#' @export
dct_pool <- function(ny, nx, order = c("zigzag", "raster")) {
  checkmate::assert_count(ny, positive = TRUE)
  checkmate::assert_count(nx, positive = TRUE)
  order <- match.arg(order)
  structure(
    list(type = "dct", ny = as.integer(ny), nx = as.integer(nx), order = order),
    class = "imfeatures_pooler"
  )
}

#' Radial centre/periphery pooling
#'
#' @param breaks Increasing numeric breakpoints in \eqn{[0, 1]} for normalized
#'   radius from the spatial centre. The default
#'   \code{c(0, 0.25, 0.5, 1)} yields three annuli.
#' @export
radial_pool <- function(breaks = c(0, 0.25, 0.5, 1)) {
  checkmate::assert_numeric(breaks, finite = TRUE, any.missing = FALSE, min.len = 2)
  if (is.unsorted(breaks, strictly = TRUE)) {
    stop("'breaks' must be strictly increasing.")
  }
  structure(
    list(type = "radial", breaks = as.numeric(breaks)),
    class = "imfeatures_pooler"
  )
}

#' Leave activations unchanged, flattening 4-D maps
#' @export
identity_pool <- function() {
  structure(list(type = "identity"), class = "imfeatures_pooler")
}

#' Apply a pooler to an activation tensor
#'
#' @param pooler An \code{imfeatures_pooler}.
#' @param a Numeric matrix (\code{N × P}) or 4-D array.
#' @param layout Tensor layout for 4-D inputs; see \code{\link{as_nhwc}}.
#' @param ... Unused.
#'
#' @return An \code{N × P} numeric matrix.
#' @export
pool_activations <- function(pooler, a, layout = "nhwc", ...) {
  UseMethod("pool_activations")
}

#' @export
pool_activations.imfeatures_pooler <- function(pooler, a, layout = "nhwc", ...) {
  a <- as_nhwc(a, layout = layout)
  d <- dim(a)
  if (is.null(d) || length(d) == 2L) {
    if (identical(pooler$type, "identity") || length(d) == 2L) {
      m <- as.matrix(a)
      storage.mode(m) <- "double"
      return(m)
    }
    stop("Pooler '", pooler$type, "' expects a 4-D activation tensor.")
  }
  if (length(d) != 4L) {
    stop("Activations must be a matrix or a 4-D array.")
  }
  switch(pooler$type,
    gap = .pool_global(a, fun = mean),
    gmp = .pool_global(a, fun = max),
    grid = .pool_grid(a, pooler$ny, pooler$nx),
    dct = .pool_dct(a, pooler$ny, pooler$nx, pooler$order),
    radial = .pool_radial(a, pooler$breaks),
    identity = .pool_identity_4d(a),
    stop("Unknown pooler type: ", pooler$type)
  )
}

#' Feature names for a pooled stage
#'
#' @param pooler An \code{imfeatures_pooler}.
#' @param stage Stage name used as the first token.
#' @param n_channels Number of channels, or the number of dense features when
#'   \code{spatial} is \code{NULL} and the pooler is identity.
#' @param spatial Optional integer length-2 spatial size \code{c(H, W)} for
#'   identity flattening of 4-D maps.
#' @export
pooler_feature_names <- function(pooler, stage, n_channels, spatial = NULL) {
  if (!inherits(pooler, "imfeatures_pooler")) {
    stop("'pooler' must be created by one of the pooler constructors.")
  }
  n_channels <- as.integer(n_channels)
  switch(pooler$type,
    gap =,
    gmp = vapply(
      seq_len(n_channels),
      function(c) .stage_feature_name(stage, c, pooler$type, n_channels),
      character(1)
    ),
    grid = {
      suffixes <- sprintf(
        "grid_y%d_x%d",
        rep(seq_len(pooler$ny), each = pooler$nx),
        rep(seq_len(pooler$nx), times = pooler$ny)
      )
      .names_by_channel(stage, n_channels, suffixes)
    },
    dct = {
      pairs <- .dct_mode_pairs(pooler$ny, pooler$nx, pooler$order)
      suffixes <- sprintf("dct_y%d_x%d", pairs$v, pairs$u)
      .names_by_channel(stage, n_channels, suffixes)
    },
    radial = {
      n_rings <- length(pooler$breaks) - 1L
      suffixes <- sprintf("radial_r%d", seq_len(n_rings))
      .names_by_channel(stage, n_channels, suffixes)
    },
    identity = {
      if (is.null(spatial)) {
        vapply(
          seq_len(n_channels),
          function(i) sprintf("%s::f_%03d", stage, i),
          character(1)
        )
      } else {
        checkmate::assert_integerish(spatial, len = 2, lower = 1)
        h <- as.integer(spatial[[1]])
        w <- as.integer(spatial[[2]])
        suffixes <- sprintf(
          "y%d_x%d",
          rep(seq_len(h), times = w),
          rep(seq_len(w), each = h)
        )
        .names_by_channel(stage, n_channels, suffixes)
      }
    },
    stop("Unknown pooler type: ", pooler$type)
  )
}

#' @export
print.imfeatures_pooler <- function(x, ...) {
  label <- switch(x$type,
    gap = "Global average pool",
    gmp = "Global max pool",
    grid = sprintf("Adaptive grid pool %d x %d", x$ny, x$nx),
    dct = sprintf("DCT pool %d x %d (%s)", x$ny, x$nx, x$order),
    radial = sprintf("Radial pool (%s)", paste(x$breaks, collapse = ", ")),
    identity = "Identity pool",
    x$type
  )
  cat(label, "\n", sep = "")
  invisible(x)
}

#' @keywords internal
.names_by_channel <- function(stage, n_channels, suffixes) {
  as.character(unlist(lapply(seq_len(n_channels), function(c) {
    vapply(
      suffixes,
      function(s) .stage_feature_name(stage, c, s, n_channels),
      character(1)
    )
  }), use.names = FALSE))
}

#' @keywords internal
.pool_global <- function(a, fun) {
  out <- apply(a, c(1L, 4L), fun)
  storage.mode(out) <- "double"
  out
}

#' @keywords internal
.pool_identity_4d <- function(a) {
  n <- dim(a)[1]
  h <- dim(a)[2]
  w <- dim(a)[3]
  c <- dim(a)[4]
  tmp <- aperm(a, c(2L, 3L, 4L, 1L))
  t(matrix(tmp, nrow = h * w * c, ncol = n))
}

#' @keywords internal
.bin_index <- function(n, n_bins) {
  as.integer(cut(seq_len(n), breaks = n_bins, labels = FALSE, include.lowest = TRUE))
}

#' @keywords internal
.pool_grid <- function(a, ny, nx) {
  n <- dim(a)[1]
  h <- dim(a)[2]
  w <- dim(a)[3]
  c <- dim(a)[4]
  if (ny > h || nx > w) {
    stop(sprintf("Grid %d x %d exceeds spatial size %d x %d.", ny, nx, h, w))
  }
  y_bin <- .bin_index(h, ny)
  x_bin <- .bin_index(w, nx)
  out <- matrix(0, n, ny * nx * c)
  col <- 0L
  for (ci in seq_len(c)) {
    for (iy in seq_len(ny)) {
      for (ix in seq_len(nx)) {
        col <- col + 1L
        cell <- a[, y_bin == iy, x_bin == ix, ci, drop = FALSE]
        out[, col] <- apply(cell, 1L, mean)
      }
    }
  }
  storage.mode(out) <- "double"
  out
}

#' @keywords internal
.dct_1d_basis <- function(n, n_modes) {
  x <- seq_len(n) - 1L
  k <- seq_len(n_modes) - 1L
  alpha <- ifelse(k == 0L, 1 / sqrt(n), sqrt(2 / n))
  basis <- cos(pi * outer(k, 2 * x + 1) / (2 * n))
  sweep(basis, 1L, alpha, `*`)
}

#' @keywords internal
.dct_mode_pairs <- function(ny, nx, order = c("zigzag", "raster")) {
  order <- match.arg(order)
  if (identical(order, "raster")) {
    return(list(
      v = rep(seq_len(ny) - 1L, each = nx),
      u = rep(seq_len(nx) - 1L, times = ny)
    ))
  }
  pairs_v <- integer(0)
  pairs_u <- integer(0)
  for (s in 0:(ny + nx - 2L)) {
    if (s %% 2L == 0L) {
      v <- min(s, ny - 1L)
      u <- s - v
      while (v >= 0L && u < nx) {
        pairs_v <- c(pairs_v, v)
        pairs_u <- c(pairs_u, u)
        v <- v - 1L
        u <- u + 1L
      }
    } else {
      u <- min(s, nx - 1L)
      v <- s - u
      while (u >= 0L && v < ny) {
        pairs_v <- c(pairs_v, v)
        pairs_u <- c(pairs_u, u)
        u <- u - 1L
        v <- v + 1L
      }
    }
  }
  list(v = as.integer(pairs_v), u = as.integer(pairs_u))
}

#' @keywords internal
.dct_2d_basis <- function(h, w, ny, nx, order) {
  if (ny > h || nx > w) {
    stop(sprintf("DCT modes (%d x %d) exceed spatial size (%d x %d).", ny, nx, h, w))
  }
  pairs <- .dct_mode_pairs(ny, nx, order)
  by <- .dct_1d_basis(h, ny)
  bx <- .dct_1d_basis(w, nx)
  n_modes <- length(pairs$v)
  basis <- matrix(0, n_modes, h * w)
  for (i in seq_len(n_modes)) {
    mode <- outer(by[pairs$v[i] + 1L, ], bx[pairs$u[i] + 1L, ])
    basis[i, ] <- as.vector(mode)
  }
  if (pairs$v[1] == 0L && pairs$u[1] == 0L) {
    basis[1, ] <- basis[1, ] / sqrt(h * w)
  }
  basis
}

#' @keywords internal
.pool_dct <- function(a, ny, nx, order) {
  n <- dim(a)[1]
  h <- dim(a)[2]
  w <- dim(a)[3]
  c <- dim(a)[4]
  basis <- .dct_2d_basis(h, w, ny, nx, order)
  tmp <- aperm(a, c(2L, 3L, 1L, 4L))
  spatial <- matrix(tmp, nrow = h * w, ncol = n * c)
  coeffs <- basis %*% spatial
  arr <- array(coeffs, dim = c(nrow(basis), n, c))
  out <- matrix(aperm(arr, c(2L, 1L, 3L)), nrow = n)
  storage.mode(out) <- "double"
  out
}

#' @keywords internal
.pool_radial <- function(a, breaks) {
  n <- dim(a)[1]
  h <- dim(a)[2]
  w <- dim(a)[3]
  c <- dim(a)[4]
  cy <- (h + 1) / 2
  cx <- (w + 1) / 2
  yy <- row(matrix(0, h, w))
  xx <- col(matrix(0, h, w))
  r <- sqrt((yy - cy)^2 + (xx - cx)^2)
  r_max <- max(r)
  if (r_max > 0) {
    r <- r / r_max
  }
  ring <- findInterval(r, breaks, rightmost.closed = TRUE)
  n_rings <- length(breaks) - 1L
  out <- matrix(0, n, n_rings * c)
  col <- 0L
  for (ci in seq_len(c)) {
    for (ri in seq_len(n_rings)) {
      col <- col + 1L
      mask <- ring == ri
      if (!any(mask)) {
        out[, col] <- 0
      } else {
        slice <- a[, , , ci, drop = FALSE]
        vals <- matrix(slice, nrow = n, ncol = h * w)
        out[, col] <- rowMeans(vals[, as.vector(mask), drop = FALSE])
      }
    }
  }
  storage.mode(out) <- "double"
  out
}
