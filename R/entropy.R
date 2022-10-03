
#' @keywords internal
entropy <- function(a) {
  if (sum(a) !=1.0 && sum(a)>0) {
    a = a / sum(a)
  }
  v = a>0.0
  -sum(a[v] * log2(a[v]))
}

#' @keywords internal
first_order_entropy <- function(fres, gabor_bins) {
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
  img <- imager::load.image(file)
  if (!is.null(max_pixels)) {
    isize <- dim(img)[1:2]

    #a = np.sqrt(max_pixels / float(img.size[0]*img.size[1]))
    #img = img.resize((int(img.size[0]*a),int(img.size[1]*a)), Image.ANTIALIAS)

    a = sqrt(max_pixels / (isize[1]*isize[2]))
    img <- imager::resize(img, as.integer(isize[1]*a), as.integer(isize[2]*a),interpolation_type=6)
  }

  image_raw = as.array(imager::grayscale(img)) # luma transform
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
  for (i in 1:num_filters) {
    img_filt[i,,] <- imager::convolve(iraw, imager::as.cimg(fbank$flt_weights[i,,]))
  }

  resp_bin <- apply(img_filt, c(2,3), which.max)
  #resp_bin = np.argmax(self.image_flt, axis=0)
  resp_val <- apply(img_filt, c(2,3), max)

  resp_val <- zero_borders(resp_val, 2)

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
  resp_val[1:nlines,] <- 0
  resp_val[, 1:nlines] <- 0
  resp_val[(nr-nlines):nr,] <- 0
  resp_val[, (nc-nlines):nc] <- 0
  resp_val
}

#' @keywords internal
do_counting <- function(fres, maxdiag=80, circbins=48) {
  isize <- fres$fimg$image_size
  w <- isize[1]
  h <- isize[2]


  #normalize_fac = float(filter_img.resp_val.shape[0]*filter_img.resp_val.shape[1])
  #complex_before = np.sum(filter_img.resp_val)/normalize_fac

  resp_val <- fres$resp_val

  # cutoff minor filter responses
  normalize_fac = dim(resp_val)[1]*dim(resp_val)[2]
  complex_before = sum(resp_val)/normalize_fac


  cutoff = sort(as.vector(resp_val), decreasing=TRUE)[10000] # get 10000th highest response for cutting of beneath
  resp_val[resp_val<cutoff] = 0
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
    distance_rel[distance_rel>=max_diagonal] = max_diagonal

    direction <- round(atan2(ey-ey[cp], ex-ex[cp]) / (2.0*pi)*circ_bins + (orientations[cp]/gabor_bins*circ_bins))
    direction <- (direction+circ_bins) %% circ_bins
    ind <- cbind(distance_rel, direction+1, orientations_rel+1)
    counts[ind] <- counts[ind] + resp_val[cbind(ex,ey)] * resp_val[ex[cp],ey[cp]]
    #np.add.at(counts, cbind(distance_rel, direction, orientations_rel),
    #          fres.resp_val[cbind(ey,ex)] * fres.resp_val[ey[cp],ex[cp])
  }

  list(counts=counts, complex_before=complex_before)
}


#' @keywords internal
do_statistics <- function(counts, bins_vec) {

  #counts_sum = sum(counts, axis=2) + 0.00001
  counts_sum <- apply(counts, c(1,2), sum) + .00001
  normalized_counts <- sweep(counts, c(1,2), counts_sum, "/")
  #normalized_counts <- counts / (counts_sum[,,,np.newaxis])

  x = normalized_counts * cos(bins_vec)
  y = normalized_counts * sin(bins_vec)
  #mean_vector = mean(x+1i*y, axis=2)
  mean_vector <- apply(x+1i*y, c(1,2), mean)
  #circular_mean_angle = np.mod(np.angle(mean_vector) + 2*np.pi, 2*np.pi)
  circular_mean_angle <- (Arg(mean_vector) + 2*pi) %% (2*pi)
  circular_mean_length = abs(mean_vector)

  # correction as proposed by Zar 1999
  d = 2*pi/gabor_bins
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

  list(normalized_counts=normalized_counts,
       circular_mean_angle=circular_mean_angle,
       circular_mean_length=circular_mean_length,
       shannon=shannon,
       shannon_nan=shannon_nan)
}



#' @export
edge_entropy <- function(impath, max_pixels=150*200, gabor_bins=24, filter_length=31) {
  fimg <- filtered_image(impath, max_pixels)
  fbank <- filter_bank(gabor_bins, filter_length)
  fres <- run_filterbank(fimg, fbank)
  cts <- do_counting(fres)
  stats <- do_statistics(cts$counts, fres$fbank$bins_vec)

  ranges <- list(c(20,80), c(80,160), c(160,240))
  fo <- first_order_entropy(fres)

  shannon <- lapply(ranges, function(r) {
    mean(rowMeans(stats$shannon)[r[1]:r[2]])
  })

  data.frame(im=impath, entropy=fo, pentropy_20_80=shannon[[1]], pentropy_80_160=shannon[[2]], pentropy_160_240=shannon[[3]],
             complex_before=cts$complex_before)

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
