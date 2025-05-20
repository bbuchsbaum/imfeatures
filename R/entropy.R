#' @useDynLib imfeatures, .registration = TRUE
#' @importFrom Rcpp sourceCpp
NULL

#' @keywords internal
entropy <- function(a) {
  # Ensure input is numeric
  if (!is.numeric(a)) {
    stop("`a` must be numeric")
  }

  # Remove NA values
  a <- a[!is.na(a)]

  if (length(a) == 0) {
    warning("All values are NA")
    return(NA_real_)
  }

  s <- sum(a)
  if (s == 0) {
    warning("Sum of probabilities is zero")
    return(NA_real_)
  }

  eps <- sqrt(.Machine$double.eps)
  if (abs(s - 1) > eps) {
    a <- a / s
  }

  v <- a > 0.0
  -sum(a[v] * log2(a[v]))
}

#' @keywords internal
first_order_entropy <- function(fres, gabor_bins = NULL) {
  first_order_bin = numeric(fres$num_filters)
  for (b in 1:fres$num_filters) {
    first_order_bin[b] = sum(fres$resp_val[fres$resp_bin==b])
  }
  first_order = entropy(first_order_bin)
  first_order
}


#' @keywords internal
filter_bank <- function(num_filters, flt_size,  octave=3) {

  bins_vec = head(seq(0, 2*pi, length.out=num_filters+1), -1)

  flt_raw = array(0, c(num_filters, flt_size, flt_size))
  for (i in 1:num_filters) {
    #filter_bank.set_flt(i, filter_bank.create_gabor(FILTER_SIZE, theta=BINS_VEC[i], octave=3))
    flt_raw[i,,] <- create_gabor(flt_size, theta=bins_vec[i], octave=octave)
    # --- Debug: Print range/sum of the first filter --- 
    if (i == 1) {
      message("[R] filter_bank: First Gabor filter range: [", 
              min(flt_raw[i,,]), ", ", max(flt_raw[i,,]), 
              "], Sum = ", sum(flt_raw[i,,]))
    }
    # -----------------------------------------------
  }

  ret <- list(bins_vec=bins_vec,
       flt_weights=flt_raw,
       octave=octave,
       flt_size=flt_size)

  class(ret) <- c("filter_bank")
  ret
}

#' @keywords internal
create_gaussian <- function(size, sigma=2) {
    valsy = seq(-size/2+1, size/2, length.out=size)
    valsx = seq(-size/2+1, size/2, length.out=size)
    mg <- pracma::meshgrid(valsx, valsy)
    xgr <- mg$X
    ygr <- mg$Y
    gaussian <- exp(-(xgr^2 + ygr^2)/(2*sigma*sigma))
    gaussian/sum(gaussian)
}

#' @keywords internal
create_gabor <- function(size, theta, octave=3) {
  amplitude = 1.0
  phase = pi/2.0
  frequency = 0.5^octave # 0.5**(octave+0.5)
  hrsf = 4 # half response spatial frequency bandwith
  sigma = 1/(pi*frequency) * sqrt(log(2)/2) * (2.0^hrsf+1)/(2.0^hrsf-1)
  valsy = seq(-size/2+1, size/2, length.out=size)
  valsx = seq(-size/2+1, size/2, length.out=size)
  mg <- pracma::meshgrid(valsx, valsy)
  xgr <- mg$X
  ygr <- mg$Y

  omega = 2*pi*frequency
  gaussian = exp(-(xgr*xgr + ygr*ygr)/(2*sigma*sigma))
  slant = xgr*(omega*sin(theta)) + ygr*(omega*cos(theta))
  gabor = gaussian * amplitude*cos(slant + phase);
  gabor

}

#' @keywords internal
filtered_image <- function(file, max_pixels=300*300) {
  # Validate input file
  if (!is.character(file) || length(file) != 1) {
    stop("'file' must be a single character string (file path)")
  }
  
  if (!file.exists(file)) {
    stop("Image file not found: ", file)
  }
  
  # Try to load the image and handle errors gracefully
  img <- tryCatch({
    imager::load.image(file)
  }, error = function(e) {
    stop("Failed to load image: ", file, "\nError: ", e$message)
  })
  
  # Verify it's a valid image object
  if (!inherits(img, "cimg")) {
    stop("imager::load.image did not return a valid cimg object for: ", file)
  }
  
  if (!is.null(max_pixels)) {
    isize <- dim(img)[1:2]

    a = sqrt(max_pixels / (isize[1]*isize[2]))
    img <- tryCatch({
      imager::resize(img, as.integer(isize[1]*a), as.integer(isize[2]*a), interpolation_type=6)
    }, error = function(e) {
      warning("Image resize failed, using original size. Error: ", e$message)
      img # Return original image
    })
  }

  # Convert to grayscale and ensure it's a valid matrix
  image_raw = tryCatch({
    as.array(imager::grayscale(img)) # luma transform
  }, error = function(e) {
    stop("Failed to convert image to grayscale: ", e$message)
  })
  
  if (!is.array(image_raw) || length(dim(image_raw)) < 2) {
    stop("Failed to create valid image matrix from: ", file)
  }
  
  ret <- list(file=file, img=img, image_raw=image_raw, image_size=dim(image_raw)[1:2])
  class(ret) <- c("filtered_image", "list")
  ret
}


#' @keywords internal
run_filterbank <- function(fimg, fbank) {

  shape <- dim(fimg$image_raw)[1:2]
  h <- shape[1]
  w <- shape[2]
  num_filters <- length(fbank$bins_vec)
  img_filt <- array(0, c(num_filters, h, w))

  iraw <- imager::as.cimg(fimg$image_raw)
  # --- Debug: Print input image range --- 
  message("[R] run_filterbank: Input image range: [", 
          min(iraw), ", ", max(iraw), 
          "], Mean = ", mean(iraw))
  # ------------------------------------
  
  for (i in 1:num_filters) {
    conv_result <- imager::convolve(iraw, imager::as.cimg(fbank$flt_weights[i,,]))
    img_filt[i,,] <- as.array(conv_result) # Ensure it's an array for min/max
    # --- Debug: Print range/sum of the first convolution result --- 
    if (i == 1) {
        message("[R] run_filterbank: First convolution result (conv) range: [", 
                min(img_filt[i,,]), ", ", max(img_filt[i,,]), 
                "], Sum = ", sum(img_filt[i,,]))
    }
    # -----------------------------------------------------------
  }

  resp_bin <- apply(img_filt, c(2,3), which.max)
  #resp_bin = np.argmax(self.image_flt, axis=0)
  resp_val <- apply(img_filt, c(2,3), max)

  resp_val <- zero_borders(resp_val, 2)

  message("[R] run_filterbank: Range resp_val (after border zero): [", 
          min(resp_val), ", ", max(resp_val), "]")

  ret <- list(
    fimg=fimg,
    fbank=fbank,
    img_filt=img_filt,
    num_filters=num_filters,
    resp_bin=resp_bin,
    resp_val=resp_val)

  class(ret) <- "filtered_set"
  ret
}

#' @keywords internal
zero_borders <- function(resp_val, nlines=2) {
  nr <- nrow(resp_val)
  nc <- ncol(resp_val)

  # Clamp nlines to valid range and ensure non-negative
  nlines <- max(0, min(nlines, nr - 1, nc - 1))

  if (nlines > 0) {
    resp_val[1:nlines, ] <- 0
    resp_val[, 1:nlines] <- 0
    resp_val[(nr - nlines + 1):nr, ] <- 0
    resp_val[, (nc - nlines + 1):nc] <- 0
  }

  resp_val
}

#' @keywords internal
do_counting <- function(fres, maxdiag=80, circ_bins=48) {
  isize <- fres$fimg$image_size
  w <- isize[1]
  h <- isize[2]

  #normalize_fac = float(filter_img.resp_val.shape[0]*filter_img.resp_val.shape[1])
  #complex_before = np.sum(filter_img.resp_val)/normalize_fac

  resp_val <- fres$resp_val
  message("[R] do_counting: Range resp_val (input): [", 
          min(resp_val), ", ", max(resp_val), "]")

  # cutoff minor filter responses
  normalize_fac = dim(resp_val)[1]*dim(resp_val)[2]

  ## gradient magnitude
  complex_before = sum(resp_val)/normalize_fac


  # Determine the k-th largest response to use as cutoff. For small images
  # length(resp_val) may be < 10,000, so we clamp k to the number of pixels.
  k <- min(10000L, length(resp_val))
  cutoff = sort(as.vector(resp_val), decreasing = TRUE)[k]
  # If k equals the number of pixels the cutoff will be the minimum response
  # value, meaning no responses are removed.
  message(sprintf("[R] do_counting: Calculated cutoff (k=%d): %f", k, cutoff))
  resp_val[resp_val<cutoff] = 0
  message("[R] do_counting: Range resp_val (after cutoff): [", 
          min(resp_val), ", ", max(resp_val), "]")
  #ey, ex = filter_img.resp_val.nonzero()

  # lookup tables to speed up calculations
  edge_dims = dim(resp_val)
  mg <- pracma::meshgrid(seq(-edge_dims[1], edge_dims[1], length.out=2*edge_dims[1]+1),
                         seq(-edge_dims[2], edge_dims[2], length.out=2*edge_dims[2]+1))
  xx <- mg$X
  yy <- mg$Y

  dist = t(sqrt(xx^2+yy^2))
  ecds <- which(resp_val != 0, arr.ind=TRUE)

  ex <- ecds[,1]
  ey <- ecds[,2]

  orientations = fres$resp_bin[cbind(ex,ey)]

  counts = array(0, c(maxdiag, circ_bins, length(fres$fbank$bins_vec)))
  gabor_bins <- length(fres$fbank$bins_vec)
  #print "Counting", filter_img.image_name, filter_img.image_size(), "comparing", ex.size
  for (cp in 1:length(ex)) {
    #print(cp)

    orientations_rel = orientations - orientations[cp]
    orientations_rel = (orientations_rel + gabor_bins) %% gabor_bins

    i1 <- (ex-ex[cp])+edge_dims[1]
    i2 <- (ey-ey[cp])+edge_dims[2]
    distance_rel = round(dist[cbind(i1,i2)]) + 1
    distance_rel[distance_rel>=maxdiag] = maxdiag

    direction <- round(atan2(ey-ey[cp], ex-ex[cp]) / (2.0*pi)*circ_bins + (orientations[cp]/gabor_bins*circ_bins))
    direction <- (direction+circ_bins) %% circ_bins
    ind <- cbind(distance_rel, direction+1, orientations_rel+1)
    counts[ind] <- counts[ind] + resp_val[cbind(ex,ey)] * resp_val[ex[cp],ey[cp]]
    #np.add.at(counts, cbind(distance_rel, direction, orientations_rel),
    #          fres.resp_val[cbind(ey,ex)] * fres.resp_val[ey[cp],ex[cp])
  }

  message("[R] do_counting: Range counts cube: [", 
          min(counts), ", ", max(counts), 
          "], Sum = ", sum(counts))

  list(counts=counts, complex_before=complex_before)
}


#' @keywords internal
do_statistics <- function(counts, bins_vec) {

  #counts_sum = sum(counts, axis=2) + 0.00001
  counts_sum <- apply(counts, c(1,2), sum) + .00001
  message("[R] do_statistics: counts_sum range: [", 
          min(counts_sum), ", ", max(counts_sum), 
          "], Sum = ", sum(counts_sum))
          
  normalized_counts <- sweep(counts, c(1,2), counts_sum, "/")
  message("[R] do_statistics: normalized_counts range: [", 
          min(normalized_counts), ", ", max(normalized_counts), 
          "], Sum = ", sum(normalized_counts)) # Sum should be ~ maxdiag * circ_bins
  #normalized_counts <- counts / (counts_sum[,,,np.newaxis])

  x = normalized_counts * cos(bins_vec)
  y = normalized_counts * sin(bins_vec)
  #mean_vector = mean(x+1i*y, axis=2)
  mean_vector <- apply(x+1i*y, c(1,2), mean)
  #circular_mean_angle = np.mod(np.angle(mean_vector) + 2*np.pi, 2*np.pi)
  circular_mean_angle <- (Arg(mean_vector) + 2*pi) %% (2*pi)
  circular_mean_length = abs(mean_vector)

  # correction as proposed by Zar 1999
  d = 2*pi/length(bins_vec)
  c = d / 2.0 / sin(d/2.0)
  circular_mean_length = circular_mean_length * c

  d <- dim(normalized_counts)[1]
  a <- dim(normalized_counts)[2]
  #d,a,_ = normalized_counts.shape
  shannon = matrix(0, d,a)
  shannon_nan = matrix(0, d,a)
  for (di in seq(1,d)) {
    for (ai in seq(1,a)) {
      shannon[di,ai] = entropy(normalized_counts[di,ai,])
      if (counts_sum[di,ai]>1) {
        shannon_nan[di,ai] = shannon[di,ai]
      } else {
        shannon_nan[di,ai] = NaN
      }
    }
  }

  message("[R] do_statistics: shannon matrix range: [", 
          min(shannon, na.rm=TRUE), ", ", max(shannon, na.rm=TRUE), "]")

  list(normalized_counts=normalized_counts,
       circular_mean_angle=circular_mean_angle,
       circular_mean_length=circular_mean_length,
       shannon=shannon,
       shannon_nan=shannon_nan)
}



#' Calculate Edge Entropy Features from Images
#'
#' This function calculates first-order and pairwise edge entropy features from an image,
#' which can be used for analyzing texture and structural complexity in images.
#'
#' @param image Either a file path to an image (character string) or a numeric matrix
#'        representing a grayscale image. If a file path is provided, the image will
#'        be loaded and converted to grayscale.
#' @param max_pixels Integer. The maximum number of pixels allowed in the processed image.
#'        Larger images will be resized. Only used when \code{image} is a file path.
#'        Set to NULL to disable resizing. Defaults to 300*400.
#' @param maxdiag Integer. Maximum diagonal distance for pairwise entropy calculations.
#'        Defaults to 500.
#' @param gabor_bins Integer. Number of orientation bins for Gabor filter bank.
#'        Defaults to 24.
#' @param filter_length Integer. Size of the Gabor filters (must be odd). 
#'        Defaults to 31.
#' @param circ_bins Integer. Number of circular bins for directional statistics.
#'        Defaults to 48.
#' @param ranges List of integer vectors. Each vector should contain two elements
#'        specifying the start and end indices for grouping pairwise entropy at
#'        different distance ranges. Defaults to list(c(20,80), c(80,160), c(160,240)).
#' @param use_cpp Logical. Whether to use the C++ implementation (generally faster).
#'        Defaults to TRUE. If FALSE, uses the pure R implementation.
#'
#' @return A data frame with the following columns:
#'   \item{im}{Image identifier (file path or "matrix" for matrix input)}
#'   \item{entropy}{First-order entropy value}
#'   \item{pentropy_20_80}{Pairwise entropy for distance range 20-80}
#'   \item{pentropy_80_160}{Pairwise entropy for distance range 80-160}
#'   \item{pentropy_160_240}{Pairwise entropy for distance range 160-240}
#'   \item{complex_before}{Image complexity measure before thresholding}
#'
#' @details
#' Edge entropy measures quantify the distribution and organization of oriented edges
#' in images. The method applies a bank of Gabor filters at different orientations
#' and measures both first-order entropy (distribution of dominant orientations) and
#' pairwise entropy (how orientation relationships vary with distance and direction).
#'
#' The C++ implementation is substantially faster for larger images but requires the same inputs.
#' It automatically converts the ranges list to the required format for the C++ function.
#'
#' @examples
#' \donttest{
#' # Using a file path
#'
#'
#' # Using a matrix
#' img_matrix <- matrix(runif(100*100), nrow=100)
#' result <- edge_entropy(img_matrix)
#' }
#'
#' @export
edge_entropy <- function(image, max_pixels=300*400, maxdiag=500, gabor_bins=24,
                         filter_length=31, circ_bins=48, 
                         ranges=list(c(20,80), c(80, 160), c(160,240)),
                         use_cpp=TRUE) {
  
  # Check if image is a file path or a matrix
  if (is.character(image) && length(image) == 1) {
    # It's a file path
    impath <- image
    
    # Check if the file exists
    if (!file.exists(impath)) {
      stop("Image file does not exist: ", impath)
    }
    
    if (use_cpp) {
      # For C++ implementation, we load the image and convert to matrix here
      fimg <- tryCatch({
        filtered_image(impath, max_pixels)
      }, error = function(e) {
        stop("Failed to load image: ", impath, "\nError: ", e$message)
      })
      
      # Verify image_raw is a matrix
      if (!is.matrix(fimg$image_raw) && !is.array(fimg$image_raw)) {
        stop("Failed to convert image to matrix: ", impath)
      }
      
      image_matrix <- fimg$image_raw
      # Additional check for numeric matrix with valid dimensions
      if (!is.numeric(image_matrix) || length(dim(image_matrix)) < 2) {
        stop("Image must be a numeric matrix/array with at least 2 dimensions")
      }
      
      result <- edge_entropy_cpp(
        image = as.matrix(image_matrix),
        impath = impath,
        maxdiag = as.integer(maxdiag),
        gabor_bins = as.integer(gabor_bins),
        filter_length = as.integer(filter_length),
        circ_bins = as.integer(circ_bins),
        ranges = lapply(ranges, function(r) as.integer(r))
      )
      return(result)
    } else {
      # Use the existing R implementation
      fimg <- filtered_image(impath, max_pixels)
      fbank <- filter_bank(gabor_bins, filter_length)
      fres <- run_filterbank(fimg, fbank)
      cts <- do_counting(fres, maxdiag=maxdiag, circ_bins=circ_bins)
      stats <- do_statistics(cts$counts, fres$fbank$bins_vec)
      
      fo <- first_order_entropy(fres)
      
      shannon_summary <- lapply(ranges, function(r) {
        # Ensure rowMeans are calculated correctly and handle potential NaNs
        valid_rows <- r[1]:r[2]
        # Check bounds
        valid_rows <- valid_rows[valid_rows <= nrow(stats$shannon)]
        if (length(valid_rows) == 0) return(NA)
        
        row_means_subset <- rowMeans(stats$shannon[valid_rows, , drop = FALSE], na.rm = TRUE)
        # Handle case where all values in a range might be NaN or empty
        mean_val <- mean(row_means_subset, na.rm = TRUE)
        if (!is.finite(mean_val)) mean_val <- NA
        
        message(sprintf("[R] edge_entropy: Range [%d, %d]: rowmeans mean = %.6f", r[1], r[2], mean_val))
        mean_val
      })
      
      message("[R] edge_entropy: Calculated fo = ", fo)
      message("[R] edge_entropy: final_shannon[1] = ", shannon_summary[[1]])
      message("[R] edge_entropy: final_shannon[2] = ", shannon_summary[[2]])
      message("[R] edge_entropy: final_shannon[3] = ", shannon_summary[[3]])
      
      return(data.frame(
        im=impath, 
        entropy=fo, 
        pentropy_20_80=shannon_summary[[1]], 
        pentropy_80_160=shannon_summary[[2]], 
        pentropy_160_240=shannon_summary[[3]],
        complex_before=cts$complex_before
      ))
    }
  } else if (is.matrix(image)) {
    # It's a matrix
    
    # Add additional validation for the matrix
    if (!is.numeric(image)) {
      stop("Image matrix must contain numeric values")
    }
    
    if (length(dim(image)) != 2) {
      stop("Image must be a 2D matrix, not higher-dimensional array")
    }
    
    if (nrow(image) < 3 || ncol(image) < 3) {
      stop("Image matrix is too small (minimum size: 3x3)")
    }
    
    if (use_cpp) {
      result <- tryCatch({
        edge_entropy_cpp(
          image = image,
          impath = "matrix", # Use "matrix" as the identifier
          maxdiag = as.integer(maxdiag),
          gabor_bins = as.integer(gabor_bins),
          filter_length = as.integer(filter_length),
          circ_bins = as.integer(circ_bins),
          ranges = lapply(ranges, function(r) as.integer(r))
        )
      }, error = function(e) {
        stop("C++ function failed: ", e$message)
      })
      return(result)
    } else {
      # Create a filtered_image object from the matrix
      fimg <- list(
        file = "matrix",
        img = NULL, # No imager object for direct matrix input
        image_raw = image,
        image_size = dim(image)
      )
      class(fimg) <- c("filtered_image", "list")
      
      # Then proceed with the R implementation
      fbank <- filter_bank(gabor_bins, filter_length)
      fres <- run_filterbank(fimg, fbank)
      cts <- do_counting(fres, maxdiag=maxdiag, circ_bins=circ_bins)
      stats <- do_statistics(cts$counts, fres$fbank$bins_vec)
      
      fo <- first_order_entropy(fres)
      
      shannon_summary <- lapply(ranges, function(r) {
        # Ensure rowMeans are calculated correctly and handle potential NaNs
        valid_rows <- r[1]:r[2]
        # Check bounds
        valid_rows <- valid_rows[valid_rows <= nrow(stats$shannon)]
        if (length(valid_rows) == 0) return(NA)
        
        row_means_subset <- rowMeans(stats$shannon[valid_rows, , drop = FALSE], na.rm = TRUE)
        # Handle case where all values in a range might be NaN or empty
        mean_val <- mean(row_means_subset, na.rm = TRUE)
        if (!is.finite(mean_val)) mean_val <- NA
        
        message(sprintf("[R] edge_entropy (matrix): Range [%d, %d]: rowmeans mean = %.6f", r[1], r[2], mean_val))
        mean_val
      })
      
      message("[R] edge_entropy (matrix): Calculated fo = ", fo)
      message("[R] edge_entropy (matrix): final_shannon[1] = ", shannon_summary[[1]])
      message("[R] edge_entropy (matrix): final_shannon[2] = ", shannon_summary[[2]])
      message("[R] edge_entropy (matrix): final_shannon[3] = ", shannon_summary[[3]])
      
      return(data.frame(
        im="matrix", 
        entropy=fo, 
        pentropy_20_80=shannon_summary[[1]], 
        pentropy_80_160=shannon_summary[[2]], 
        pentropy_160_240=shannon_summary[[3]],
        complex_before=cts$complex_before
      ))
    }
  } else {
    # Not a file path or matrix
    if (is.null(image)) {
      stop("'image' cannot be NULL. Must be a file path or numeric matrix")
    } else if (is.character(image) && length(image) > 1) {
      stop("'image' must be a single file path, not a character vector")
    } else if (is.data.frame(image)) {
      stop("'image' is a data.frame. Please convert to a matrix with as.matrix()")
    } else {
      stop("'image' must be either a file path (character string) or a numeric matrix, not ", class(image)[1])
    }
  }
}

#filter_bank = Filter_bank(GABOR_BINS, flt_size=FILTER_SIZE)
#for i in range(filter_bank.num_filters):
#  filter_bank.set_flt(i, filter_bank.create_gabor(FILTER_SIZE, theta=BINS_VEC[i], octave=3))

#img = FilterImage(file_list[i], max_pixels=MAX_PIXELS)
#img.run_filterbank(filter_bank)
#counts, complex_before = do_counting(img, os.path.basename(file_list[i]))

#fimg <- filtered_image("testdata2/ballerina.jpg", 200*100)
#fbank <- filter_bank(24, 31)
#fres <- run_filterbank(fimg, fbank)
#cts <- do_counting(fres)
#stats <- do_statistics(cts$counts, fres$fbank$bins_vec)
