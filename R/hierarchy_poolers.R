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

#' Spatial poolers for convolutional activations
#'
#' A pooler turns a 4-D activation tensor (\code{N} images x \code{H} x
#' \code{W} x \code{C} channels) into an \code{N x P} feature matrix. The
#' constructors below only describe \emph{how} to pool;
#' \code{\link{pool_activations}} does the work, and
#' \code{\link{vgg_hierarchy_spec}} assigns one pooler to each network stage.
#'
#' \tabular{lll}{
#'   \strong{Constructor} \tab \strong{Columns per channel} \tab \strong{What is kept} \cr
#'   \code{global_average_pool()} \tab 1 \tab Mean over all positions \cr
#'   \code{global_max_pool()} \tab 1 \tab Maximum over all positions \cr
#'   \code{adaptive_grid_pool(ny, nx)} \tab \code{ny * nx} \tab Mean in each cell of a coarse grid \cr
#'   \code{dct_pool(ny, nx)} \tab \code{ny * nx} \tab Mean plus low spatial frequencies \cr
#'   \code{radial_pool(breaks)} \tab \code{length(breaks) - 1} \tab Mean in each centre-to-periphery ring \cr
#'   \code{identity_pool()} \tab \code{H * W} \tab Every position (flattened)
#' }
#'
#' Every pooler writes its columns channel by channel: all spatial features
#' of channel 1, then all of channel 2, and so on.
#' \code{\link{pooler_feature_names}} returns the matching column names.
#'
#' Dense activations (\code{N x P} matrices, such as \code{fc1} or logits)
#' have no spatial axes. Every pooler returns them unchanged.
#'
#' @section Choosing a pooler:
#' \itemize{
#'   \item Use \code{global_average_pool()} when you want one number per
#'     channel and do not care where in the image a feature occurred.
#'   \item Use \code{global_max_pool()} for the same size when presence of a
#'     feature matters more than its extent.
#'   \item Use \code{adaptive_grid_pool()} to keep coarse location in
#'     non-negative, directly interpretable cells.
#'   \item Use \code{dct_pool()} to keep coarse layout as a mean plus smooth
#'     spatial contrasts. The first column per channel equals global average
#'     pooling, so the remaining columns add layout to the usual summary.
#'   \item Use \code{radial_pool()} when eccentricity (centre versus
#'     periphery) is the spatial variable of interest.
#'   \item Use \code{identity_pool()} for dense layers, or for small maps
#'     where you want every position.
#' }
#'
#' @return An object of class \code{imfeatures_pooler}: a list holding the
#'   pooler \code{type} and its parameters.
#'
#' @examples
#' # Two images, an 8 x 8 map, three channels (N x H x W x C)
#' a <- array(rnorm(2 * 8 * 8 * 3), dim = c(2, 8, 8, 3))
#'
#' dim(pool_activations(global_average_pool(), a))    # 2 x 3
#' dim(pool_activations(adaptive_grid_pool(2, 2), a)) # 2 x 12
#' dim(pool_activations(dct_pool(3, 3), a))           # 2 x 27
#' dim(pool_activations(radial_pool(), a))            # 2 x 9
#' dim(pool_activations(identity_pool(), a))          # 2 x 192
#'
#' # The first DCT column of each channel is the global average
#' all.equal(
#'   pool_activations(dct_pool(3, 3), a)[, 1],
#'   pool_activations(global_average_pool(), a)[, 1]
#' )
#'
#' @seealso \code{\link{pool_activations}}, \code{\link{pooler_feature_names}},
#'   \code{\link{vgg_hierarchy_spec}}, and
#'   \code{vignette("spatial-pooling", package = "imfeatures")}.
#' @name poolers
#' @aliases imfeatures_pooler
NULL

#' @describeIn poolers Mean over height and width; one column per channel.
#' @export
global_average_pool <- function() {
  structure(list(type = "gap"), class = "imfeatures_pooler")
}

#' @describeIn poolers Maximum over height and width; one column per channel.
#' @export
global_max_pool <- function() {
  structure(list(type = "gmp"), class = "imfeatures_pooler")
}

#' Adaptive rectangular grid pooling
#'
#' Averages each channel within the cells of an \code{ny x nx} grid laid over
#' the activation map. The grid adapts to the map: the same pooler works on a
#' 224 x 224 map and on a 14 x 14 map.
#'
#' Each spatial axis is cut into contiguous, non-overlapping bins that
#' together cover every position. When the map size is a multiple of the
#' number of bins, all bins have the same size and the mean of the cells
#' equals global average pooling. Otherwise the bins differ in size (14
#' positions in 4 bins gives 4, 3, 3, 4), and the mean of the cells is only
#' close to the global average. This differs from PyTorch's
#' \code{AdaptiveAvgPool2d}, which lets neighbouring windows overlap in that
#' case.
#'
#' Columns are ordered by channel, then grid row (top to bottom), then grid
#' column (left to right). Names take the form
#' \code{stage::channel_001::grid_y1_x1}, with one-based cell indices.
#'
#' @param ny,nx Number of bins along height and width. Neither may exceed the
#'   size of the activation map it is applied to.
#'
#' @return An \code{imfeatures_pooler} that yields \code{ny * nx} columns per
#'   channel.
#'
#' @examples
#' # One image, a 4 x 4 map, one channel, with a bright top-right quadrant
#' a <- array(0, dim = c(1, 4, 4, 1))
#' a[1, 1:2, 3:4, 1] <- 1
#' out <- pool_activations(adaptive_grid_pool(2, 2), a)
#' colnames(out) <- pooler_feature_names(adaptive_grid_pool(2, 2), "b5", 1)
#' out
#'
#' @seealso \code{\link{poolers}} for the other pooling options.
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
#' Projects each channel's activation map onto the lowest \code{ny x nx}
#' modes of a two-dimensional discrete cosine transform (DCT-II). The result
#' summarizes \emph{where} a channel is active as a mean plus a few smooth
#' spatial contrasts (left versus right, top versus bottom, centre versus
#' edges, and so on).
#'
#' Mode \code{(v, u)} has \code{v} half-cycles along height and \code{u}
#' half-cycles along width. All \code{ny * nx} modes with \code{v < ny} and
#' \code{u < nx} are kept.
#'
#' @section Scale of the coefficients:
#' The mode \code{(0, 0)} is rescaled to be the spatial mean, so the first
#' column of each channel equals \code{\link{global_average_pool}}. All
#' other modes use the orthonormal DCT-II basis. Their coefficients grow with
#' the square root of the number of positions: a unit-amplitude cosine
#' pattern gives a coefficient near 9.9 on a 14 x 14 map and near 158 on a
#' 224 x 224 map. Columns from different modes or different stages are
#' therefore not on a common scale. Standardize the columns before you
#' combine them. \code{\link{block_pca}} does so by default
#' (\code{feature_scale = "sd"}).
#'
#' @section Column order and names:
#' Columns are ordered by channel, then by mode. \code{order = "zigzag"}
#' sorts modes from low to high frequency along anti-diagonals, as in JPEG:
#' \code{(0,0), (0,1), (1,0), (2,0), (1,1), (0,2), ...}. \code{order =
#' "raster"} sorts them row by row. Both orders keep the same modes; only
#' the column order differs. Names take the form
#' \code{stage::channel_001::dct_y0_x1}, where \code{y} and \code{x} are the
#' zero-based frequencies \code{v} and \code{u}.
#'
#' @param ny,nx Number of retained DCT modes along height and width. Neither
#'   may exceed the size of the activation map it is applied to.
#' @param order \code{"zigzag"} (default) or \code{"raster"}.
#'
#' @return An \code{imfeatures_pooler} that yields \code{ny * nx} columns per
#'   channel.
#'
#' @examples
#' # A channel that is active only on the left half of a 6 x 6 map
#' a <- array(0, dim = c(1, 6, 6, 1))
#' a[1, , 1:3, 1] <- 1
#' out <- pool_activations(dct_pool(2, 2), a)
#' colnames(out) <- pooler_feature_names(dct_pool(2, 2), "b3", 1)
#' round(out, 3)
#' # dct_y0_x0 is the mean (0.5); dct_y0_x1 carries the left-right contrast.
#'
#' @seealso \code{\link{poolers}} for the other pooling options, and
#'   \code{\link{vgg16_block_end_spec}} for a ready-made VGG-16 spec that
#'   uses DCT pooling.
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
#' Averages each channel within concentric rings around the centre of the
#' activation map. Use it when distance from the image centre matters and
#' direction does not.
#'
#' Radius is measured from the geometric centre of the map and divided by
#' the distance to the corners, so 0 is the centre and 1 is a corner. On a
#' square map the midpoint of each edge lies at about 0.71. Ring \code{k}
#' holds positions with \code{breaks[k] <= radius < breaks[k + 1]}; the last
#' ring also includes its upper break. Positions outside the range of
#' \code{breaks} are left out, so \code{breaks = c(0, 0.5)} pools the central
#' region only.
#'
#' Rings cover unequal numbers of positions. With the default breaks on a
#' 14 x 14 map, the three rings hold 16, 52, and 128 positions. On small
#' maps a ring can hold no positions at all; its columns are then 0 and a
#' warning is raised. The default breaks need a map of about 5 x 5 or larger:
#' on a 4 x 4 map no position lies within 0.25 of the centre, because the
#' centre falls between positions.
#'
#' Columns are ordered by channel, then ring (centre outwards). Names take
#' the form \code{stage::channel_001::radial_r1}.
#'
#' @param breaks Strictly increasing numeric breakpoints in \eqn{[0, 1]} for
#'   normalized radius from the spatial centre. The default
#'   \code{c(0, 0.25, 0.5, 1)} yields three rings.
#'
#' @return An \code{imfeatures_pooler} that yields
#'   \code{length(breaks) - 1} columns per channel.
#'
#' @examples
#' # A channel that is active only in the central 2 x 2 positions
#' a <- array(0, dim = c(1, 8, 8, 1))
#' a[1, 4:5, 4:5, 1] <- 1
#' out <- pool_activations(radial_pool(), a)
#' colnames(out) <- pooler_feature_names(radial_pool(), "b4", 1)
#' round(out, 3)
#'
#' @seealso \code{\link{poolers}} for the other pooling options.
#' @export
radial_pool <- function(breaks = c(0, 0.25, 0.5, 1)) {
  checkmate::assert_numeric(
    breaks,
    lower = 0, upper = 1, finite = TRUE, any.missing = FALSE, min.len = 2
  )
  if (is.unsorted(breaks, strictly = TRUE)) {
    stop("'breaks' must be strictly increasing.")
  }
  structure(
    list(type = "radial", breaks = as.numeric(breaks)),
    class = "imfeatures_pooler"
  )
}

#' @describeIn poolers No pooling. Dense matrices pass through unchanged;
#'   4-D maps are flattened to \code{H * W * C} columns ordered by channel,
#'   then width, then height (height varies fastest).
#' @export
identity_pool <- function() {
  structure(list(type = "identity"), class = "imfeatures_pooler")
}

#' Apply a pooler to an activation tensor
#'
#' Reduces a batch of activations to one feature row per image. This is the
#' function \code{\link{extract_feature_hierarchy}} calls for each stage; you
#' can also call it directly on activations from any backend.
#'
#' A 4-D array is first put in \code{N x H x W x C} order (see
#' \code{\link{as_nhwc}}) and then pooled over height and width. A 2-D
#' \code{N x P} matrix has no spatial axes and is returned unchanged by every
#' pooler, which is how dense layers pass through a spec.
#'
#' @param pooler An \code{imfeatures_pooler}; see \code{\link{poolers}}.
#' @param a Numeric matrix (\code{N x P}) or 4-D array with one image per
#'   slice of the first dimension.
#' @param layout Tensor layout for 4-D inputs: \code{"nhwc"} (Keras, default)
#'   or \code{"nchw"} (PyTorch). See \code{\link{as_nhwc}}.
#' @param ... Unused.
#'
#' @return An \code{N x P} numeric matrix without column names. \code{P} is
#'   the number of channels times the pooler's columns per channel. Use
#'   \code{\link{pooler_feature_names}} to label the columns.
#'
#' @examples
#' # PyTorch-style activations: 2 images, 3 channels, 8 x 8 map
#' nchw <- array(rnorm(2 * 3 * 8 * 8), dim = c(2, 3, 8, 8))
#' feats <- pool_activations(adaptive_grid_pool(2, 2), nchw, layout = "nchw")
#' colnames(feats) <- pooler_feature_names(adaptive_grid_pool(2, 2), "layer4", 3)
#' dim(feats)
#' head(colnames(feats))
#'
#' @seealso \code{\link{poolers}}, \code{\link{pooler_feature_names}}.
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
#' Builds the column names that match the output of
#' \code{\link{pool_activations}}, in the same order. Names have three
#' tokens separated by \code{"::"}: the stage, the zero-padded channel, and
#' the spatial feature, for example \code{b3::channel_017::dct_y0_x1}.
#'
#' \tabular{ll}{
#'   \strong{Pooler} \tab \strong{Spatial token} \cr
#'   \code{global_average_pool()} \tab \code{gap} \cr
#'   \code{global_max_pool()} \tab \code{gmp} \cr
#'   \code{adaptive_grid_pool()} \tab \code{grid_y<row>_x<col>} (one-based cell) \cr
#'   \code{dct_pool()} \tab \code{dct_y<v>_x<u>} (zero-based frequency) \cr
#'   \code{radial_pool()} \tab \code{radial_r<ring>} (one-based, centre outwards) \cr
#'   \code{identity_pool()} on a 4-D map \tab \code{y<row>_x<col>} (one-based position) \cr
#'   \code{identity_pool()} on a dense layer \tab \code{stage::f_001} (two tokens)
#' }
#'
#' @param pooler An \code{imfeatures_pooler}.
#' @param stage Stage name used as the first token.
#' @param n_channels Number of channels, or the number of dense features when
#'   \code{spatial} is \code{NULL} and the pooler is identity.
#' @param spatial Optional integer length-2 spatial size \code{c(H, W)} for
#'   identity flattening of 4-D maps.
#'
#' @return A character vector with one name per pooled column.
#'
#' @examples
#' pooler_feature_names(global_average_pool(), "b5", 2)
#' pooler_feature_names(dct_pool(2, 2), "b5", 1)
#' pooler_feature_names(identity_pool(), "b5", 1, spatial = c(2, 2))
#' pooler_feature_names(identity_pool(), "fc1", 3)
#'
#' @seealso \code{\link{pool_activations}}, \code{\link{poolers}}.
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
  # Contiguous bin for each of n positions. Bin k holds positions i with
  # (k - 1) * (n - 1) / n_bins < i - 1 <= k * (n - 1) / n_bins, and position 1
  # joins bin 1. This is what cut(seq_len(n), breaks = n_bins) returns, done
  # in integer arithmetic so that it also covers a single bin and cannot
  # shift with floating-point rounding of the break points.
  n <- as.integer(n)
  n_bins <- as.integer(n_bins)
  if (n_bins == 1L || n == 1L) {
    return(rep.int(1L, n))
  }
  i <- seq_len(n) - 1L
  pmax(1L, (i * n_bins + n - 2L) %/% (n - 1L))
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
  empty <- logical(n_rings)
  col <- 0L
  for (ci in seq_len(c)) {
    for (ri in seq_len(n_rings)) {
      col <- col + 1L
      mask <- ring == ri
      if (!any(mask)) {
        out[, col] <- 0
        empty[ri] <- TRUE
      } else {
        slice <- a[, , , ci, drop = FALSE]
        vals <- matrix(slice, nrow = n, ncol = h * w)
        out[, col] <- rowMeans(vals[, as.vector(mask), drop = FALSE])
      }
    }
  }
  if (any(empty)) {
    warning(sprintf(
      "Radial ring(s) %s hold no positions on a %d x %d map; their columns are 0.",
      paste(which(empty), collapse = ", "), h, w
    ), call. = FALSE)
  }
  storage.mode(out) <- "double"
  out
}
